#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r @"..\packages\FSharp.Charting.0.90.13\lib\net40\FSharp.Charting.dll"
#r @"..\packages\MathNet.Numerics.3.10.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\DotNumerics.1.1\lib\DotNumerics.dll"


#load "..\packages\MathNet.Numerics.FSharp.3.10.0\MathNet.Numerics.fsx"
#load "..\packages\FSharp.Charting.0.90.13\FSharp.Charting.fsx"
#load @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\EarthDrawn.FSharp.ML.Source\Common.fs"

open System
open FSharp.Data
open FSharp.Charting
open MathNet.Numerics.LinearAlgebra
open Common

// list of tuples (int * int) that are the indicies of the 
// train, cross validation and testing data
let indices = Common.getIndicies rawData.RowCount 

// Build training data
let X_train = Common.getSubSetOfMatrix features (indices.[0])
let y_train = Common.getSubSetOfMatrix classifications (indices.[0])

let iterations = topolgy.Length

// sigmoid
let sigmoid (z: Matrix<float>): Matrix<float> = 
        z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))

// FOR  POST
let α = 0.01
let initialTheta = Matrix.Build.Dense(X_train.ColumnCount, 1, 0.0)

// perform ones step of gradient descent 
let descent (theta:Matrix<float>) =
    let m      = (float X_train.RowCount)
    let hx     = sigmoid (X_train*theta)
    let h      = hx-y_train 
    let delt_J = X_train 
                    |> Matrix.mapRows (fun i row -> h.[i, 0]*row)
                    |> Matrix.sumCols
                    |> Matrix.Build.DenseOfRowVectors
    let regTerm = theta 
                    |> Matrix.mapi(fun i j y_i -> if (i<>0) then y_i else 0.0) 
                    |> Matrix.sum
    theta - (α*((1.0/m) * delt_J.Transpose()) + (λ/m*regTerm))

// recursively applies descent function
let rec gradientDescent (count: int) (gradAccum:Matrix<float>) =
    if count = 0 then
        gradientDescent (count+1) (gradAccum.Append(descent initialTheta)) 
    elif count < iterations then
        let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count))
        gradientDescent (count+1) (gradAccum.Append(descent prevTheta))
    else
        gradAccum
    
