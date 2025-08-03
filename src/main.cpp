#include "pennis.hpp"
#include <cstdlib>
#include <chrono>

int main()
{
    try
    {
        std::vector<uint32_t> layerSizes = {1, 4, 4, 4, 1};
        std::vector<uint32_t> actTypes = {Tanh, Tanh, Tanh, Tanh};
        AdamParams adamParams;

        PENNIS pennis(256, layerSizes, actTypes, adamParams);
        
        const int epochs = 5000;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> input, target;

        for(int epoch = 0; epoch < epochs; epoch++)
        {
            for (int i = 0; i < 10; i++)
            {
                float x = float(i) / 9 * 3.141593 * 2;

                std::vector<float> input = {x};
                std::vector<float> target = {sin(x)};
                pennis.uploadInputs(input);
                pennis.uploadTargets(target);
                pennis.runForward();
                pennis.runBackprop();
                pennis.applyAdam();
            }

            if(epoch % 100 == 0 || epoch == epochs - 1)
            {
                std::cout << "\nEpoch " << epoch << ":";
                pennis.printArchitecture();
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "Training Time: " << duration.count() << " ms\n";
        std::cout << "Inference After Training \n";
        for (int i = 0; i < 100; i++)
        {
            float x = float(i) / 99 * 3.141593 * 2;
            std::vector<float> input = {x};
            auto output = pennis.predict(input);
            std::cout << "(" << input[0] << ", " << output[0] << "), ";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}