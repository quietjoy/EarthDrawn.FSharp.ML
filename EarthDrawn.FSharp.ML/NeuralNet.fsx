
let Jo (theta: float) = 2.0*(theta**4.0)+2.0

let theta = 1.0
let e = 0.01

let theta1 = theta + e
let theta2 = theta - e

let res = ((Jo theta1) - (Jo theta2))/(2.0*e)