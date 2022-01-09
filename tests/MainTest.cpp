#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE ManifGeomCppTests

#include <iostream>
#include <boost/test/unit_test.hpp>

/*! Global testing definitions. */
struct GlobalFixture
{
    GlobalFixture()
    {
        // common function calls here
        srand(444444);
    }

    ~GlobalFixture() {}
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
