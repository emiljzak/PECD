Quadrature rules taken from: https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/sphere_lebedev_rule.html

SPHERE_LEBEDEV_RULE is a dataset directory which contains files defining Lebedev rules on the unit sphere, which can be used for quadrature, and have a known precision.

A Lebedev rule of precision p can be used to correctly integrate any polynomial for which the highest degree term xiyjzk satisfies i+j+k <= p.

The approximation to the integral of f(x) has the form Integral f(x,y,z) = 4 * pi * sum ( 1 <= i < n ) wi * f(xi,yi,zi) where

        xi = cos ( thetai ) * sin ( phii )
        yi = sin ( thetai ) * sin ( phii )
        zi =                  cos ( phii )

The data file for an n point rule includes n lines, where the i-th line lists the values of

        thetai phii wi

The angles are measured in degrees, and chosen so that:
        - 180 <= thetai <= + 180
            0 <= phii <= + 180

and the weights wi should sum to 1.
Licensing:

The computer code and data files described and made available on this web page are distributed under the GNU LGPL license.

Related Data and Programs:

GEOMETRY, a FORTRAN90 library which computes various geometric quantities, including grids on spheres.

SCVT, a FORTRAN90 library which can find a set of well separated points on a sphere using Centroidal Voronoi Tessellations.

SPHERE_DESIGN_RULE, a dataset directory which contains files defining point sets on the surface of the unit sphere, known as "designs", which can be useful for estimating integrals on the surface, among other uses.

SPHERE_DESIGN_RULE, a FORTRAN90 library which returns point sets on the surface of the unit sphere, known as "designs", which can be useful for estimating integrals on the surface, among other uses.

SPHERE_GRID, a dataset directory which contains grids of points, lines, triangles or quadrilaterals on a sphere;

SPHERE_GRID, a FORTRAN90 library which provides a number of ways of generating grids of points, or of points and lines, or of points and lines and faces, over the unit sphere.

SPHERE_VORONOI, a MATLAB program which computes the Voronoi diagram of points on a sphere.

SPHERE_VORONOI_DISPLAY_OPENGL, a C++ program which displays a sphere and randomly selected generator points, and then gradually colors in points in the sphere that are closest to each generator.

SPHERE_XYZ_DISPLAY, a MATLAB program which reads XYZ information defining points in 3D, and displays a unit sphere and the points in the MATLAB graphics window.

SPHERE_XYZ_DISPLAY_OPENGL, a C++ program which reads XYZ information defining points in 3D, and displays a unit sphere and the points, using OpenGL.

Reference:

Thomas Ericson, Victor Zinoviev,
Codes on Euclidean Spheres,
Elsevier, 2001,
ISBN: 0444503293,
LC: QA166.7E75
Gerald Folland,
How to Integrate a Polynomial Over a Sphere,
American Mathematical Monthly,
Volume 108, Number 5, May 2001, pages 446-448.
Vyacheslav Lebedev, Dmitri Laikov,
A quadrature formula for the sphere of the 131st algebraic order of accuracy,
Russian Academy of Sciences Doklady Mathematics,
Volume 59, Number 3, 1999, pages 477-481.
Vyacheslav Lebedev,
A quadrature formula for the sphere of 59th algebraic order of accuracy,
Russian Academy of Sciences Doklady Mathematics,
Volume 50, 1995, pages 283-286.
Vyacheslav Lebedev, A.L. Skorokhodov,
Quadrature formulas of orders 41, 47, and 53 for the sphere,
Russian Academy of Sciences Doklady Mathematics,
Volume 45, 1992, pages 587-592.
Vyacheslav Lebedev,
Spherical quadrature formulas exact to orders 25-29,
Siberian Mathematical Journal,
Volume 18, 1977, pages 99-107.
Vyacheslav Lebedev,
Quadratures on a sphere,
Computational Mathematics and Mathematical Physics,
Volume 16, 1976, pages 10-24.
Vyacheslav Lebedev,
Values of the nodes and weights of ninth to seventeenth order Gauss-Markov quadrature formulae invariant under the octahedron group with inversion,
Computational Mathematics and Mathematical Physics,
Volume 15, 1975, pages 44-51.
AD McLaren,
Optimal Numerical Integration on a Sphere,
Mathematics of Computation,
Volume 17, Number 84, October 1963, pages 361-383.
Edward Saff, Arno Kuijlaars,
Distributing Many Points on a Sphere,
The Mathematical Intelligencer,
Volume 19, Number 1, 1997, pages 5-11.
Sample files

lebedev_003.txt, 6 point rule, precision 3.
lebedev_005.txt, 14 point rule, precision 5.
lebedev_007.txt, 26 point rule, precision 7.
lebedev_009.txt, 38 point rule, precision 9.
lebedev_011.txt, 50 point rule, precision 11.
lebedev_013.txt, 74 point rule, precision 13.
lebedev_015.txt, 86 point rule, precision 15.
lebedev_017.txt, 110 point rule, precision 17.
lebedev_019.txt, 146 point rule, precision 19.
lebedev_021.txt, 170 point rule, precision 21.
lebedev_023.txt, 194 point rule, precision 23.
lebedev_025.txt, 230 point rule, precision 25.
lebedev_027.txt, 266 point rule, precision 27.
lebedev_029.txt, 302 point rule, precision 29.
lebedev_031.txt, 350 point rule, precision 31.
lebedev_035.txt, 434 point rule, precision 35.
lebedev_041.txt, 590 point rule, precision 41.
lebedev_047.txt, 770 point rule, precision 47.
lebedev_053.txt, 974 point rule, precision 53.
lebedev_059.txt, 1202 point rule, precision 59.
lebedev_065.txt, 1454 point rule, precision 65.
lebedev_071.txt, 1730 point rule, precision 71.
lebedev_077.txt, 2030 point rule, precision 77.
lebedev_083.txt, 2354 point rule, precision 83.
lebedev_089.txt, 2702 point rule, precision 89.
lebedev_095.txt, 3074 point rule, precision 95.
lebedev_101.txt, 3470 point rule, precision 101.
lebedev_107.txt, 3890 point rule, precision 107.
lebedev_113.txt, 4334 point rule, precision 113.
lebedev_119.txt, 4802 point rule, precision 119.
lebedev_125.txt, 5294 point rule, precision 125.
lebedev_131.txt, 5810 point rule, precision 131.
You can go up one level to the DATASETS page.

Last revised on 09 September 2010.
