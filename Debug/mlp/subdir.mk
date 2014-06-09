################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../mlp/layer.cpp \
../mlp/mlp.cpp \
../mlp/neuron.cpp 

OBJS += \
./mlp/layer.o \
./mlp/mlp.o \
./mlp/neuron.o 

CPP_DEPS += \
./mlp/layer.d \
./mlp/mlp.d \
./mlp/neuron.d 


# Each subdirectory must supply rules for building sources it contributes
mlp/%.o: ../mlp/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


