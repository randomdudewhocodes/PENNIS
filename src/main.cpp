#include "pennis.hpp"
#include <cstdlib>
#include <chrono>

int main()
{
    try
    {
        std::vector<uint32_t> layerSizes = {1, 4, 4, 4, 1};
        std::vector<uint32_t> actTypes = {Tanh, Tanh, Tanh, None};
        AdamParams adamParams = {0.9f, 0.999f, 1e-8f, 0.01f};

        PENNIS pennis(50, layerSizes, actTypes, adamParams);
        
        const int epochs = 5000;

        auto start_time = std::chrono::high_resolution_clock::now();

        std::vector<float> input, target;

        for (int i = 0; i < 50; i++)
        {
            float x = float(i) / 49 * 3.141593 * 2;

            input.push_back(x);
            target.push_back(sin(x));
        }

        pennis.uploadInputs(input);
        pennis.uploadTargets(target);

        for(int epoch = 0; epoch < epochs; epoch++)
        {
            pennis.train();

            //if(epoch % 100 == 0 || epoch == epochs - 1)
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