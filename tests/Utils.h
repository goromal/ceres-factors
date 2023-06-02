#include <Eigen/Core>
#include <chrono>
#include <SO2.h>
#include <SE2.h>
#include <SO3.h>
#include <SE3.h>

using namespace Eigen;

namespace test_utils
{
void getNoiselessRangeBearing2DData(const SE2d&     x,
                                    const Vector2d& l,
                                    double&         d,
                                    double&         theta,
                                    Vector2d&       p,
                                    double&         phi);
} // namespace test_utils
