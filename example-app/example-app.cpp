#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
	if (argc != 2)  {
		std::cerr << "Usage: example-app <path-to-export-module>\n";
		return -1;
	}

	torch::jit::script::Module module;

	try {
		module = torch::jit::load(argv[1]);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::ones({1, 3, 224, 224}));
		at::Tensor output = module.forward(inputs).toTensor();
		std::cout << output.slice(/*dim*/1,/*start*/ 0, /*end*/5) << "\n";
	} catch (const c10::Error& e) {
		std::cerr << "error loading the model \n";
		return -1;
	}
	std::cout << "ok\n";
}
