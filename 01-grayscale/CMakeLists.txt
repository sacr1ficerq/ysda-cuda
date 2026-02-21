add_catch(test_grayscale test.cpp grayscale.cuh grayscale.cu)
target_link_libraries(test_grayscale PRIVATE ysda_cuda::util stb::stb_image)
set_target_properties(test_grayscale PROPERTIES CUDA_ARCHITECTURES "50;75;86")
