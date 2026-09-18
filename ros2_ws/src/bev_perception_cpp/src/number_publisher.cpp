#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/int32.hpp"

using namespace std::chrono_literals;

class NumberPublisher : public rclcpp::Node
{
public:
  NumberPublisher()
  : Node("number_publisher"), count_(0)
  {
    publisher_ =
      this->create_publisher<std_msgs::msg::Int32>("number", 10);

    timer_ =
      this->create_wall_timer(
        1s,
        std::bind(&NumberPublisher::publish_number, this)
      );
  }

private:
  void publish_number()
  {
    std_msgs::msg::Int32 message;
    message.data = count_++;

    RCLCPP_INFO(
      this->get_logger(),
      "Publishing: %d",
      message.data
    );

    publisher_->publish(message);
  }

  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  int count_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::spin(
    std::make_shared<NumberPublisher>()
  );

  rclcpp::shutdown();

  return 0;
}