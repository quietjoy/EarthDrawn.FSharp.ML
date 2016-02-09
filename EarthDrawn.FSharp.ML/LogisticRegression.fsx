#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r @"..\packages\MathNet.Numerics.3.10.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\DotNumerics.1.1\lib\DotNumerics.dll"


#load "..\packages\MathNet.Numerics.FSharp.3.10.0\MathNet.Numerics.fsx"
#load @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\EarthDrawn.FSharp.ML.Source\LogisticRegression.fs"
#load @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\EarthDrawn.FSharp.ML.Source\Common.fs"

open System
open FSharp.Data
open MathNet.Numerics.LinearAlgebra
open EarthDrawn.FSharp.ML.Source
open Common

// ********************************
// USING MODULE AND TYPE
// ********************************
let path = @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\Skin_NonSkinSample.csv"

let raw = Common.readData path
let lambda = 1.0
let alpha = 0.01 
let logisiticReg = LogisticRegression.LogReg(alpha, lambda, 100, raw)

let error = logisiticReg.error

let costs = logisiticReg.costs |> Matrix.toArray2
let grads = logisiticReg.gradients |> Matrix.toColArrays

// replicating calculate cost
let theta = Matrix.Build.DenseOfColumnVectors(logisiticReg.gradients.Column(3))
let m     = (float logisiticReg.X_train.RowCount) 
//let hx    = logisiticReg.sigmoid (logisiticReg.X_train*theta)

logisiticReg.y_train |> Matrix.mapi(fun i j y_i -> (float i))

let hx     = logisiticReg.sigmoid (logisiticReg.X_train*theta)
let h      = hx-logisiticReg.y_train 
h.[1,0]
let delt_J = logisiticReg.X_train |> Matrix.mapRows (fun i row -> h.[i, 0]*row)

 