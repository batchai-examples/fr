!!!!test_begin!!!!

#include <gtest/gtest.h>
#include "selective_scan/csrc/selective_scan/static_switch.h"

// Test case for true condition
TEST(StaticSwitchTest, TrueCondition) {
    // Arrange
    constexpr bool flag = true;

    // Act
    auto result = BOOL_SWITCH(flag, BoolConst, []() { return 42; });

    // Assert
    EXPECT_EQ(result, 42);
}

// Test case for false condition
TEST(StaticSwitchTest, FalseCondition) {
    // Arrange
    constexpr bool flag = false;

    // Act
    auto result = BOOL_SWITCH(flag, BoolConst, []() { return 99; });

    // Assert
    EXPECT_EQ(result, 99);
}

// Test case for corner case: true condition with different code block
TEST(StaticSwitchTest, TrueConditionDifferentCodeBlock) {
    // Arrange
    constexpr bool flag = true;

    // Act
    auto result = BOOL_SWITCH(flag, BoolConst, []() { return "Hello"; });

    // Assert
    EXPECT_EQ(result, std::string("Hello"));
}

// Test case for corner case: false condition with different code block
TEST(StaticSwitchTest, FalseConditionDifferentCodeBlock) {
    // Arrange
    constexpr bool flag = false;

    // Act
    auto result = BOOL_SWITCH(flag, BoolConst, []() { return "World"; });

    // Assert
    EXPECT_EQ(result, std::string("World"));
}

!!!!test_end!!!!
