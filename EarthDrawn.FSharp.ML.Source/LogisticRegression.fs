namespace EarthDrawn.FSharp.ML.Source

module LogisticRegression =
    open System
    open FSharp.Data
    open MathNet.Numerics.LinearAlgebra

    type LogReg (y0:Matrix<float>, x0:Matrix<float>, a0: float, l0: float) = 
        member this.y = y0
        member this.X = x0
        member this.alpha = a0
        member this.lambda = l0

        let sigmoid z = 1.0 / (1.0 + exp -z)
