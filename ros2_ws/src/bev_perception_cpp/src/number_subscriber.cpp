#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

class NumberSubscriber : public rclcpp::Node
{
public:
  NumberSubscriber()
  : Node("number_subscriber")
  {
    subscription_ =
      this->create_subscription<std_msgs::msg::Int32>(
        "number",
        10,
        std::bind(
          &NumberSubscriber::number_callback,
          this,
          std::placeholders::_1
        )
      );
  }

private:
  void number_callback(const std_msgs::msg::Int32::SharedPtr msg)
  {
    RCLCPP_INFO(
      this->get_logger(),
      "Received: %d",
      msg->data
    );
  }

  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<NumberSubscriber>()
  );

  rclcpp::shutdown();
  return 0;
}