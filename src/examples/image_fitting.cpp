#include "pennis.hpp"

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

static GLuint createTexture(int width, int height, const unsigned char* data) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, data);
    return tex;
}

static void updateTexture(GLuint tex, int width, int height, const unsigned char* data) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, data);
}

int main()
{
    try
    {
        int width = 0, height = 0, channels = 0;
        const int desired_channels = 4;
        unsigned char* data = stbi_load("src/examples/image.jpg", &width, &height, &channels, desired_channels);

        if (!data)
        {
            std::cerr << "Failed to load image.jpg\n";
            return EXIT_FAILURE;
        }

        channels = desired_channels;

        const int numPixels = width * height;
        const int numSamples = numPixels;
        std::vector<unsigned char> imageData(data, data + (numPixels * channels));
        stbi_image_free(data);

        const int inputDim = 2;
        const int fourierBands = 256;
        const float fourierSigma = 10.0f;
        const bool includeInput = true;

        std::vector<uint32_t> layerSizes = { (uint32_t)inputDim, 256, 256, 256, 256, 256, (uint32_t)channels };
        std::vector<uint32_t> actTypes   = { Sine, Sine, Sine, Sine, Sine, None };
        AdamParams adamParams = { 0.9f, 0.999f, 1e-8f, 0.001f, 0.001f };

        const uint32_t workgroupSize = 512;
        const int      batchSize     = 256;
        const int      epochs        = 5000;

        std::mt19937 rng{ std::random_device{}() };

        std::vector<float> coords(numPixels * inputDim);
        #pragma omp parallel for
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int p = y * width + x;
                coords[p * inputDim + 0] = 2.0f * float(x) / width - 1.0f;
                coords[p * inputDim + 1] = 2.0f * float(y) / height - 1.0f;
            }
        }

        PENNIS net(workgroupSize, batchSize, layerSizes, actTypes, adamParams,
                   fourierBands, fourierSigma, includeInput);

        if (!glfwInit()) return -1;
        GLFWwindow* window = glfwCreateWindow(800, 600, "NN Training Visualizer", NULL, NULL);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        std::vector<unsigned char> preview(width * height * channels, 0);

        std::vector<int> previewOrder(numPixels);
        for (int i = 0; i < numPixels; ++i) previewOrder[i] = i;
        std::shuffle(previewOrder.begin(), previewOrder.end(), rng);
        int previewCursor = 0;

        int currentEpoch = 0;
        bool saved = false;
        float loss = 0.0f;

        std::uniform_int_distribution<int> dist(0, numSamples - 1);

        GLuint tex = createTexture(width, height, preview.data());

        const int trainStepsPerFrame = 4;
        const int previewChunk = 256;

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (currentEpoch < epochs)
            {
                int remaining = std::min(previewChunk, numPixels - previewCursor);
                for (int i = 0; i < remaining; ++i)
                {
                    int p = previewOrder[previewCursor + i];
                    std::vector<float> inp(inputDim);
                    std::copy_n(coords.begin() + p*inputDim, inputDim, inp.begin());
                    std::vector<float> out = net.predict(inp);
                    int idx = p * channels;
                    for (int c = 0; c < channels; ++c) {
                        float v = clamp01(out[c]);
                        preview[idx + c] = static_cast<unsigned char>(std::lrint(v * 255.0f));
                    }
                }

                previewCursor += remaining;
                if (previewCursor >= numPixels)
                {
                    std::shuffle(previewOrder.begin(), previewOrder.end(), rng);
                    previewCursor = 0;
                }
            }
            else if (!saved)
            {
                for (int p = 0; p < numPixels; ++p)
                {
                    std::vector<float> inp(inputDim);
                    std::copy_n(coords.begin() + p*inputDim, inputDim, inp.begin());
                    std::vector<float> out = net.predict(inp);
                    int idx = p * channels;
                    for (int c = 0; c < channels; ++c) {
                        float v = clamp01(out[c]);
                        preview[idx + c] = static_cast<unsigned char>(std::lrint(v * 255.0f));
                    }
                }
                updateTexture(tex, width, height, preview.data());
                net.saveArchitecture("src/examples/image_model.pnn");
                std::cout << "Training complete. Saved model.\n";
                saved = true;
            }

            updateTexture(tex, width, height, preview.data());

            ImVec2 windowSize = ImGui::GetIO().DisplaySize;
            float fontSize = ImGui::GetFontSize();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(windowSize);

            ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | 
                                     ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoSavedSettings;

            ImGui::Begin("NN Training Visualizer", nullptr, flags);
            ImGui::Text("Epoch: %d / %d", currentEpoch, epochs);
            ImGui::Text("Loss: %f", loss);
            
            ImVec2 canvasSize = ImGui::GetContentRegionAvail();

            float scale = std::min(canvasSize.x / (float)width, canvasSize.y / (float)height);

            ImGui::Image((void*)(intptr_t)tex, ImVec2(width * scale, height * scale));
            ImGui::End();

            for (int s = 0; s < trainStepsPerFrame && currentEpoch < epochs; ++s)
            {
                std::vector<int> ks(batchSize);
                for (int j = 0; j < batchSize; ++j) ks[j] = dist(rng);
                std::vector<float> in(batchSize * inputDim);
                std::vector<float> tgt(batchSize * channels);
                #pragma omp parallel for
                for (int j = 0; j < batchSize; ++j) {
                    int k = ks[j];
                    std::copy_n(coords.begin() + k*inputDim, inputDim, in.begin() + j*inputDim);
                    for (int c = 0; c < channels; ++c)
                        tgt[j*channels + c] = float(imageData[k*channels + c]) / 255.0f;
                }
                net.uploadInputs(in);
                net.uploadTargets(tgt);
                net.train();
                currentEpoch++;
            }
            loss = net.getLoss();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0,0,display_w,display_h);
            glClearColor(0.1f,0.1f,0.1f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        // cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();
    }
    catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
