#include "pennis.hpp"
#include <cstdlib>

int main()
{
    // XOR
    std::vector<float> xorInputs = {
        0.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f,
        1.0f, 1.0f
    };

    std::vector<float> xorTargets =
    {
        0.0f,
        1.0f,
        1.0f,
        0.0f
    };
    
    try
    {
        std::vector<uint32_t> layerSizes = {2, 4, 1};
        std::vector<uint32_t> actTypes = {Tanh, Sigmoid};
        AdamParams adamParams;

        PENNIS pennis(64, layerSizes, actTypes, adamParams);
        
        const int epochs = 1000;

        for(int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < 4; i++)
            {
                std::vector<float> input = {xorInputs[i * 2], xorInputs[i * 2 + 1]};
                std::vector<float> target = {xorTargets[i]};
                pennis.uploadInputs(input);
                pennis.uploadTargets(target);
                pennis.runForward();
                pennis.runBackprop();
                pennis.applyAdam();
            }

            if(epoch % 100 == 0)
            {
                std::cout << "\nEpoch " << epoch << ":";
                pennis.printArchitecture();
            }
        }

        std::cout << "Inference After Training \n";
        for (int i = 0; i < 4; i++)
        {
            std::vector<float> input = {xorInputs[i * 2], xorInputs[i * 2 + 1]};
            auto output = pennis.predict(input);
            std::cout << "Input: (" << input[0] << ", " << input[1]
                      << ") => Output: " << output[0] << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}