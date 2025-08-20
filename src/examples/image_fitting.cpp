#include "raylib.h"
#include "pennis.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

#include "stb_image.h"

static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

int main() {
    try {
        int width = 0, height = 0, channels = 0;
        const int desired_channels = 4;
        unsigned char* data = stbi_load("src/examples/image.jpg", &width, &height, &channels, desired_channels);
        if (!data) {
            std::cerr << "Failed to load image.jpg\n";
            return EXIT_FAILURE;
        }
        channels = desired_channels;

        const int numPixels = width * height;
        const int numSamples = numPixels;
        std::vector<unsigned char> imageData(data, data + (numPixels * channels));
        stbi_image_free(data);

        std::vector<uint32_t> layerSizes = { 2, 128, 128, 128, 128, 128, 128, 128, 128, (uint32_t)channels };
        std::vector<uint32_t> actTypes   = { ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, ReLU, Sigmoid };
        AdamParams adamParams = { 0.9f, 0.999f, 1e-8f, 0.001f, 0.01f };

        const uint32_t workgroupSize = 512;
        const int      batchSize     = 256;
        const int      epochs        = 5000;

        std::mt19937 rng{ std::random_device{}() };
        std::uniform_int_distribution<int> dist(0, numSamples - 1);

        PENNIS net(workgroupSize, batchSize, layerSizes, actTypes, adamParams);

        int winH = 1080;
        int winW = int(1080.0f * (float)width / (float)height);
        InitWindow(winW, winH, "Image Fitting (progressive)");
        SetTargetFPS(60);

        Image blank = GenImageColor(width, height, BLANK);
        Texture2D tex = LoadTextureFromImage(blank);
        UnloadImage(blank);

        std::vector<unsigned char> preview(width * height * channels, 0);

        std::vector<float> coords(numPixels * 2);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int p = y * width + x;
                coords[p * 2 + 0] = 2.0 * float(x) / width - 1.0f;
                coords[p * 2 + 1] = 2.0 * float(y) / height - 1.0f;
            }
        }

        std::vector<int> previewOrder(numPixels);
        for (int i = 0; i < numPixels; ++i) previewOrder[i] = i;
        std::shuffle(previewOrder.begin(), previewOrder.end(), rng);
        int previewCursor = 0;

        int currentEpoch = 0;
        bool saved = false;

        const int trainStepsPerFrame = 4;
        const int previewChunk = 256;

        float lastLoss = 0.0f;

        while (!WindowShouldClose()) {
            for (int s = 0; s < trainStepsPerFrame && currentEpoch < epochs; ++s) {
                std::vector<float> in(batchSize * 2);
                std::vector<float> tgt(batchSize * channels);

                for (int j = 0; j < batchSize; ++j) {
                    int k = dist(rng);
                    in[j * 2 + 0] = coords[k * 2 + 0];
                    in[j * 2 + 1] = coords[k * 2 + 1];
                    for (int c = 0; c < channels; ++c)
                        tgt[j * channels + c] = float(imageData[k * channels + c]) / 255.0f;
                }

                net.uploadInputs(in);
                net.uploadTargets(tgt);
                net.train();
                currentEpoch++;
            }

            lastLoss = net.getLoss();

            if (currentEpoch < epochs) {
                int remaining = std::min(previewChunk, numPixels - previewCursor);
                for (int i = 0; i < remaining; ++i) {
                    int p = previewOrder[previewCursor + i];
                    std::vector<float> out = net.predict({ coords[p * 2 + 0], coords[p * 2 + 1] });
                    int idx = p * channels;
                    for (int c = 0; c < channels; ++c) {
                        float v = clamp01(out[c]);
                        preview[idx + c] = static_cast<unsigned char>(std::lrint(v * 255.0f));
                    }
                }
                previewCursor += remaining;
                if (previewCursor >= numPixels) {
                    std::shuffle(previewOrder.begin(), previewOrder.end(), rng);
                    previewCursor = 0;
                }
            } else if (!saved) {
                // training finished -> render full texture
                for (int p = 0; p < numPixels; ++p) {
                    std::vector<float> out = net.predict({ coords[p * 2 + 0], coords[p * 2 + 1] });
                    int idx = p * channels;
                    for (int c = 0; c < channels; ++c) {
                        float v = clamp01(out[c]);
                        preview[idx + c] = static_cast<unsigned char>(std::lrint(v * 255.0f));
                    }
                }
                UpdateTexture(tex, preview.data());
                const char* filename = "image_model.pnn";
                net.saveArchitecture(filename);
                std::cout << "Training complete. Saved model to: " << filename << std::endl;
                saved = true;
            }

            UpdateTexture(tex, preview.data());

            BeginDrawing();
            ClearBackground(BLACK);

            float scale = (float)GetScreenHeight() / (float)height;
            DrawTextureEx(tex, {0,0}, 0.0f, scale, WHITE);

            DrawRectangle(8, 8, 360, 70, Fade(BLACK, 0.6f));
            DrawText(TextFormat("Epoch: %d / %d", currentEpoch, epochs), 16, 16, 18, RAYWHITE);
            DrawText(TextFormat("Loss:  %.6f", lastLoss),              16, 38, 18, RAYWHITE);
            DrawText("ESC: save & quit",                               16, 58, 18, RAYWHITE);
            EndDrawing();
        }

        UnloadTexture(tex);
        CloseWindow();

        if (!saved) {
            try {
                const char* filename = "image_model.pnn";
                net.saveArchitecture(filename);
                std::cout << "Saved model at exit to: " << filename << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "Failed to save model at exit: " << e.what() << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
