#pragma once
#include <vulkan/vulkan.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <array>
#include <optional>
#include <random>
#include <cmath>
#include <numeric>
#include <omp.h>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices
{
    std::optional<uint32_t> computeFamily;
    bool isComplete() { return computeFamily.has_value(); }
};

enum ActivationFunction { None, ReLU, Sigmoid, Tanh, Sine };

struct Buffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

struct AdamParams
{
    float beta1 = 0.9f,
          beta2 = 0.999f,
          epsilon = 1e-8f,
          learningRate = 0.001f,
          weightDecay = 0.01f;
};

struct Layer
{
    Buffer weights, biases, input, preActs, output,
           dWeightsBatch, dWeights, dBiasesBatch, dBiases, dInput, delta, dOutput,
           mWeights, vWeights, mBiases, vBiases;
    
    uint32_t inSize, outSize, actType;
};

class PENNIS
{
public:
    void uploadInputs(const std::vector<float>& inputData);
    void uploadTrainInputs(const std::vector<float>& inputData);
    void uploadTrainTargets(const std::vector<float>& targetData);
    void train();
    std::vector<float> predict(const std::vector<float>& inputData);
    std::vector<float> predictBatch(const std::vector<float>& inputData);
    void printArchitecture();
    void saveArchitecture(const std::string& filename);
    PENNIS* loadFromFile(const std::string& filename);
    float getLoss();

    PENNIS(const uint32_t workgroupSize,
           const uint32_t batchSize,
           const std::vector<uint32_t>& layerSizes,
           const std::vector<uint32_t>& activationTypes,
           const AdamParams adamParams,
           int fourierBands = 0,
           float fourierSigma = 10.0f,
           bool fourierIncludeInput = true);
    
    ~PENNIS();
private:
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue computeQueue;
    VkDescriptorSetLayout samplerDescriptorSetLayout;
    VkDescriptorSetLayout forwardDescriptorSetLayout;
    VkDescriptorSetLayout backpropDescriptorSetLayout;
    VkDescriptorSetLayout adamDescriptorSetLayout;
    VkDescriptorSetLayout reduceDescriptorSetLayout;
    VkPipelineLayout samplerPipelineLayout;
    VkPipeline samplerPipeline;
    VkPipelineLayout forwardPipelineLayout;
    VkPipeline forwardPipeline;
    VkPipelineLayout backpropPipelineLayout;
    VkPipeline backpropPipeline;
    VkPipelineLayout adamPipelineLayout;
    VkPipeline adamPipeline;
    VkPipelineLayout reducePipelineLayout;
    VkPipeline reducePipeline;
    VkCommandPool commandPool;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> samplerDescriptorSets;
    std::vector<VkDescriptorSet> forwardDescriptorSets;
    std::vector<VkDescriptorSet> backpropDescriptorSets;
    std::vector<VkDescriptorSet> adamDescriptorSets;
    std::vector<VkDescriptorSet> reduceDescriptorSets;
    VkCommandBuffer computeCommandBuffer;
    VkFence fence;
    
    uint32_t workgroupSize;
    uint32_t batchSize;
    std::vector<Layer> layers;

    Buffer trainingInputBuffer;
    Buffer trainingTargetBuffer;
    uint32_t datasetSize;
    Buffer indexBuffer;
    size_t indexCursor = 0;

    Buffer targetBuffer;
    AdamParams adamParams;
    uint32_t adamTimestep = 1;

    // Fourier feature members
    bool ffEnabled = false;
    int ff_inputDim = 0;
    int ff_numBands = 0;
    float ff_sigma = 0.0f;
    bool ff_includeInput = true;
    int ff_mappedDim = 0;
    std::vector<float> ff_B;

    void shuffleAndUploadIndices();
    void updateSamplerDescriptors();

    void initVulkan();
    void cleanup();

    void createInstance();
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createComputeDescriptorSetLayout();
    void createComputePipeline(const std::string& shaderPath,
                               VkPipelineLayout& outPipelineLayout,
                               VkPipeline& outPipeline);
    void createCommandPool();
    void createShaderStorageBuffers();
    void createDescriptorPool();
    void createComputeDescriptorSets();
    void createComputeCommandBuffers();
    void createSyncObjects();
    void recordSamplerCommandBuffer();
    void recordForwardBatchCommandBuffer();
    void recordForwardCommandBuffer();
    void recordBackpropCommandBuffer();
    void recordReduceCommandBuffer();
    void recordAdamCommandBuffer();
    void computeSubmission();
    
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    void createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties,
                      Buffer& buf);
    
    VkCommandBuffer beginSingleTimeCommands();

    void endSingleTimeCommands(VkCommandBuffer commandBuffer);

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void createDeviceLocalBufferWithStaging(const void* srcData, VkDeviceSize size, VkBufferUsageFlags dstUsage, Buffer& dstBuf);
    void uploadToDeviceBuffer(const void* srcData, VkDeviceSize size, Buffer& dstBuf);
    void readbackFromDeviceBuffer(Buffer& srcBuf, void* dstMemory, VkDeviceSize size);

    VkDescriptorBufferInfo bufferInfo(const Buffer& buf)
    {
        VkDescriptorBufferInfo info{};
        info.buffer = buf.buffer;
        info.offset = 0;
        info.range = VK_WHOLE_SIZE;

        return info;
    }

    void destroyBuffer(Buffer& buf)
    {
        if(buf.buffer != VK_NULL_HANDLE) vkDestroyBuffer(device, buf.buffer, nullptr);
        if(buf.memory != VK_NULL_HANDLE) vkFreeMemory(device, buf.memory, nullptr);

        buf.buffer = VK_NULL_HANDLE;
        buf.memory = VK_NULL_HANDLE;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error("failed to create shader module!");

        return shaderModule;
    }

    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device)
    {
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)
                indices.computeFamily = i;

            if (indices.isComplete()) break;

            i++;
        }

        return indices;
    }

    std::vector<const char*> getRequiredExtensions()
    {
        std::vector<const char*> extensions;

        if (enableValidationLayers)
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        return extensions;
    }

    bool checkValidationLayerSupport()
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers)
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) return false;
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("failed to open file!");

        size_t fileSize = (size_t) file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData)
    {
        std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }
};
