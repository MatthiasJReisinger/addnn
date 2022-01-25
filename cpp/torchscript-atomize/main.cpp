#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/script.h>

#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

static void destroyUnusedNodes(torch::jit::Graph& graph) {
    auto reversedNodesList = graph.nodes().reverse();
    std::vector<torch::jit::Node*> reversedNodes(reversedNodesList.begin(), reversedNodesList.end());

    for (int i = 0; i < reversedNodes.size(); i++) {
        if (!reversedNodes[i]->hasUses()) {
            reversedNodes[i]->destroy();
        }
    }
}

static void eliminateDeadCode(std::shared_ptr<torch::jit::Graph> graph) {
    EliminateDeadCode(graph, torch::jit::DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
    destroyUnusedNodes(*graph);
    EliminateDeadCode(graph, torch::jit::DCESideEffectPolicy::ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS);
}

template <class NamedListEntry, class NamedList>
static std::optional<NamedListEntry> findNamedListEntry(const NamedList& namedList, const std::string& name) {
    for (auto entry : namedList) {
        if (entry.name == name) {
            return entry.value;
        }
    }
    return {};
}

static std::optional<torch::jit::script::Module> findDirectChildModule(const torch::jit::script::Module& scriptModule,
                                                                       const std::string& moduleName) {
    std::optional<torch::jit::script::Module> childModule =
        findNamedListEntry<torch::jit::script::Module>(scriptModule.named_children(), moduleName);
    return childModule;
}

template <class NamedTensorList>
static std::optional<at::Tensor> findNamedTensor(const NamedTensorList& tensors, const std::string& name) {
    std::optional<at::Tensor> tensor = findNamedListEntry<at::Tensor>(tensors, name);
    return tensor;
}

static void outsourceSlice(const std::string& fileName, int startNodeIndex, int endNodeIndex,
                           std::optional<int> inputNodeIndex) {
    // reload the module from its file to ensure that we work on a fresh copy of its graph
    torch::jit::script::Module scriptModule = torch::jit::load(fileName);

    torch::jit::Method method = scriptModule.get_method("forward");
    Inline(*method.graph());
    auto nodeList = method.graph()->nodes();
    std::vector<torch::jit::Node*> nodes(nodeList.begin(), nodeList.end());

    std::cout << "outsource nodes in range [" << startNodeIndex << ":" << endNodeIndex << "]" << std::endl;

    torch::jit::Node* startNode = nodes[startNodeIndex];
    torch::jit::Node* endNode = nodes[endNodeIndex];

    if (inputNodeIndex) {
        torch::jit::Node* inputNode = nodes[*inputNodeIndex];
        inputNode->outputs()[0]->replaceAllUsesWith(method.graph()->inputs()[1]);
    }

    // use the last node's output as graph output
    while (method.graph()->outputs().size() > 0) {
        method.graph()->eraseOutput(0);
    }
    method.graph()->registerOutput(endNode->outputs()[0]);

    eliminateDeadCode(method.graph());

    // create a blank script module with an empty forward method
    torch::jit::script::Module newScriptModule("MyScriptModule");
    newScriptModule.define(R"(
            def forward(self, x):
                return x
        )");
    torch::jit::Method newMethod = newScriptModule.get_method("forward");

    std::unordered_map<std::string, torch::jit::Value*> outputMap;
    outputMap[method.graph()->inputs()[0]->debugName()] = newMethod.graph()->inputs()[0];
    outputMap[method.graph()->inputs()[1]->debugName()] = newMethod.graph()->inputs()[1];

    for (auto oldNode : method.graph()->nodes()) {
        if (oldNode->kind() == at::prim::GetAttr) {
            continue;
        }

        if (oldNode->outputs().size() != 1) {
            throw std::runtime_error("unsupported node type");
        }

        if (oldNode->kind() == at::prim::Constant) {
            auto node = newMethod.graph()->create(oldNode->kind());
            newMethod.graph()->insertNode(node);
            node->outputs()[0]->setType(oldNode->outputs()[0]->type());
            outputMap[oldNode->outputs()[0]->debugName()] = node->outputs()[0];
            node->copyAttributes(*oldNode);
        } else {
            for (auto oldInput : oldNode->inputs()) {
                auto it = outputMap.find(oldInput->debugName());
                if (it == outputMap.end()) {
                    if (oldInput->node()->kind() == at::prim::GetAttr) {
                        torch::jit::Value* inputValue = oldInput;
                        torch::jit::Node* inputNode = inputValue->node();

                        std::vector<std::string> attributeNames;
                        while (inputNode->kind() == at::prim::GetAttr) {
                            if (inputNode->inputs().size() != 1) {
                                throw std::runtime_error("unsupported input for prim::GetAttr");
                            }
                            attributeNames.push_back(inputNode->s(c10::Symbol::attr("name")));
                            inputValue = inputNode->inputs()[0];
                            inputNode = inputValue->node();
                        }

                        if (inputValue->debugName() != method.graph()->inputs()[0]->debugName()) {
                            throw std::runtime_error("unsupported input for prim::GetAttr");
                        }

                        // find the child module that holds the attribute that is referenced by the prim::GetAttr node
                        torch::jit::script::Module childModule = scriptModule;
                        for (int i = attributeNames.size() - 1; i > 0; i--) {
                            std::string childName = attributeNames[i];
                            std::optional<torch::jit::script::Module> nextChildModule =
                                findDirectChildModule(childModule, childName);
                            if (!nextChildModule) {
                                throw std::runtime_error("could not find child module");
                            }
                            childModule = *nextChildModule;
                        }

                        std::string attributeName = attributeNames[0];

                        std::optional<at::Tensor> buffer =
                            findNamedTensor(childModule.named_buffers(false), attributeName);
                        std::optional<at::Tensor> parameter =
                            findNamedTensor(childModule.named_parameters(false), attributeName);

                        std::string newAttributeName;
                        for (int i = attributeNames.size() - 1; i >= 0; i--) {
                            newAttributeName += "_" + attributeNames[i];
                        }

                        if (buffer) {
                            newScriptModule.register_buffer(newAttributeName, *buffer);
                        } else if (parameter) {
                            newScriptModule.register_parameter(newAttributeName, *parameter, false);
                        } else {
                            throw std::runtime_error("could not find tensor attribute " + attributeName);
                        }

                        auto getAttr = newMethod.graph()->create(at::prim::GetAttr);
                        newMethod.graph()->insertNode(getAttr);
                        getAttr->s_(c10::Symbol::attr("name"), newAttributeName);
                        getAttr->outputs()[0]->setType(oldInput->type());
                        getAttr->addInput(newMethod.graph()->inputs()[0]);
                        outputMap[oldInput->debugName()] = getAttr->outputs()[0];
                    } else {
                        throw std::runtime_error("unknown dependency " + oldInput->debugName());
                    }
                }
            }

            auto node = newMethod.graph()->create(oldNode->kind());
            newMethod.graph()->insertNode(node);
            node->outputs()[0]->setType(oldNode->outputs()[0]->type());
            outputMap[oldNode->outputs()[0]->debugName()] = node->outputs()[0];

            for (auto oldInput : oldNode->inputs()) {
                if (outputMap.find(oldInput->debugName()) == outputMap.end()) {
                    throw std::runtime_error("unknown dependency " + oldInput->debugName());
                } else {
                    node->addInput(outputMap[oldInput->debugName()]);
                }
            }
        }
    }

    if (outputMap.find(method.graph()->outputs()[0]->debugName()) == outputMap.end()) {
        throw std::runtime_error("unknown graph output");
    } else {
        newMethod.graph()->eraseOutput(0);
        newMethod.graph()->registerOutput(outputMap[method.graph()->outputs()[0]->debugName()]);
    }

    std::cout << "new graph: " << *newMethod.graph() << std::endl;

    std::string outFile = fileName + "_node_" + std::to_string(startNodeIndex);
    newScriptModule.save(outFile);
    std::cout << "outsourced to " << outFile << std::endl;
}

static void run(int argc, const char* argv[]) {
    if (argc != 2) {
        throw std::runtime_error("usage: torchscript-atomize <path-to-script-module>");
    }

    std::string fileName = argv[1];
    torch::jit::script::Module scriptModule;
    scriptModule = torch::jit::load(fileName);

    std::cout << "loaded " << fileName << std::endl;

    torch::jit::Method method = scriptModule.get_method("forward");

    Inline(*method.graph());

    std::unordered_multiset<std::string> unconsumedOutputs;

    if (method.graph()->inputs().size() != 2) {
        throw std::runtime_error("expect forward method with 2 arguments ('self' & a tensor argument)");
    }

    // register outgoing data-flow edges of the `forward` method's tensor argument. note that this is the second element
    // of the graph's inputs, since the first one is the `self` argument of the original `forward` method.
    torch::jit::Value* tensorInput = method.graph()->inputs()[1];
    for (auto use : tensorInput->uses()) {
        unconsumedOutputs.insert(method.graph()->inputs()[1]->debugName());
    }

    int sliceStart = 0;
    std::optional<int> sliceInput;
    int nodeIndex = 0;
    for (torch::jit::Node* node : method.graph()->nodes()) {
        // consume incoming data-flow edges
        for (torch::jit::Value* input : node->inputs()) {
            auto it = unconsumedOutputs.find(input->debugName());
            if (it != unconsumedOutputs.end()) {
                unconsumedOutputs.erase(it);
            }
        }

        if (node->kind().is_aten()) {
            if (node->outputs().size() > 1) {
                std::stringstream errorMsg;
                errorMsg << "aten node has more than one output: " << *node;
                throw std::runtime_error(errorMsg.str());
            } else if (node->outputs().size() == 0) {
                throw std::runtime_error("aten node has no outputs");
            }

            // register outgoing data-flow edges
            torch::jit::Value* output = node->outputs()[0];
            for (const torch::jit::Use& use : output->uses()) {
                unconsumedOutputs.insert(output->debugName());
            }

            if (unconsumedOutputs.size() == 1) {
                outsourceSlice(fileName, sliceStart, nodeIndex, sliceInput);
                sliceInput = nodeIndex;
                sliceStart = nodeIndex + 1;
            }
        }
        nodeIndex++;
    }
}

int main(int argc, const char* argv[]) {
    try {
        run(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
