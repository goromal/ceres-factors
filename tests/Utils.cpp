#include "Utils.h"

namespace test_utils
{
void getNoiselessRangeBearing2DData(const SE2d&     x,
                                    const Vector2d& l,
                                    double&         d,
                                    double&         theta,
                                    Vector2d&       p,
                                    double&         phi)
{
    d            = (x.t() - l).norm();
    Vector2d l_B = x.inverse() * l;
    theta        = SO2d::fromTwoUnitVectors(Vector2d(1., 0.), l_B / d).angle();
    p            = x.t();
    phi          = x.q().angle();
}
} // namespace test_utils