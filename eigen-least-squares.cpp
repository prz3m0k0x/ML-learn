#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;

int main()
{
    MatrixXd m = MatrixXd::Ones(2,2);
    auto s = m.sum();
    std::cout << s << "\n";
}