#include "raylib.h"
#include "pennis.hpp"
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cmath>
#include <iostream>

int main()
{
    try
    {
        std::vector<uint32_t> layerSizes = {1, 4, 4, 4, 4, 1};
        std::vector<uint32_t> actTypes = {Tanh, Tanh, Tanh, Tanh, None};
        AdamParams adamParams = {0.9f, 0.999f, 1e-8f, 0.01f};

        PENNIS pennis(64, 50, layerSizes, actTypes, adamParams);

        const int epochs = 5000;
        int currentEpoch = 0;

        std::vector<float> trainInput, trainTarget;
        for (int i = 0; i < 50; i++)
        {
            float x = float(i) / 49 * 2 * 3.141593f;
            trainInput.push_back(x);
            trainTarget.push_back(sinf(x));
        }

        pennis.uploadInputs(trainInput);
        pennis.uploadTargets(trainTarget);

        const int screenWidth = 800;
        const int screenHeight = 600;

        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(screenWidth, screenHeight, "NN Training Visualizer");

        const float scaleX = screenWidth / (2 * 3.141593f);
        const float scaleY = screenWidth / (2 * 3.141593f);

        std::vector<float> nnOutputs(screenWidth, 0.0f);
        std::vector<float> targetOutputs(screenWidth, 0.0f);

        for (int px = 0; px < screenWidth; px++)
        {
            float x = (float)px / scaleX;
            targetOutputs[px] = sinf(x);
        }

        while (!WindowShouldClose())
        {
            for (int i = 0; i < 100 && currentEpoch < epochs; i++)
            {
                pennis.train();
                currentEpoch++;
            }

            for (int px = 0; px < screenWidth; px++)
            {
                float x = (float)px / scaleX;
                std::vector<float> in = {x};
                auto out = pennis.predict(in);
                nnOutputs[px] = out[0];
            }

            BeginDrawing();
            ClearBackground(RAYWHITE);

            DrawLineEx({0, (float)screenHeight/2}, {(float)screenWidth, (float)screenHeight/2}, 1.5f, GRAY);

            Vector2 prevTarget;
            bool firstTarget = true;
            for (int px = 0; px < screenWidth; px++)
            {
                float py = screenHeight/2 - targetOutputs[px] * scaleY;
                Vector2 pt = {(float)px, py};

                if (!firstTarget) DrawLineEx(prevTarget, pt, 2.5f, BLUE);
                else firstTarget = false;

                prevTarget = pt;
            }

            Vector2 prevNN;
            bool firstNN = true;
            for (int px = 0; px < screenWidth; px++)
            {
                float py = screenHeight/2 - nnOutputs[px] * scaleY;
                Vector2 pt = {(float)px, py};

                if (!firstNN) DrawLineEx(prevNN, pt, 2.5f, RED);
                else firstNN = false;

                prevNN = pt;
            }

            DrawText(TextFormat("Epoch: %d / %d", currentEpoch, epochs), 10, 10, 20, BLACK);

            EndDrawing();
        }

        CloseWindow();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
