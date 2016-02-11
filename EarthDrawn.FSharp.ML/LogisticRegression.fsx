#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r @"..\packages\FSharp.Charting.0.90.13\lib\net40\FSharp.Charting.dll"
#r @"..\packages\MathNet.Numerics.3.10.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\DotNumerics.1.1\lib\DotNumerics.dll"


#load "..\packages\MathNet.Numerics.FSharp.3.10.0\MathNet.Numerics.fsx"
#load "..\packages\FSharp.Charting.0.90.13\FSharp.Charting.fsx"
#load @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\EarthDrawn.FSharp.ML.Source\LogisticRegression.fs"
#load @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\EarthDrawn.FSharp.ML.Source\Common.fs"

open System
open FSharp.Data
open FSharp.Charting
open MathNet.Numerics.LinearAlgebra
open EarthDrawn.FSharp.ML.Source
open Common


// ********************************
// USING MODULE
// ********************************
//let path = @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\Skin_NonSkinSample.csv"
//
//let raw = Common.readData path
//let lambda = 1.0
//let alpha = 10.0 
//let logisiticReg = LogisticRegression.LogReg(alpha, lambda, 100, 0.9, raw)


// ********************************
// ********************************
// ???
// ********************************
// ********************************
//let costPlot = logisiticReg.costPlot
//let error = logisiticReg.error
//let finalTheta = logisiticReg.finalTheta
//let htTheta = logisiticReg.sigmoid(logisiticReg.X_test*finalTheta) |> Matrix.toArray2
//let xx = logisiticReg.X_train |> Matrix.toArray2
//let xxx = logisiticReg.X_test |> Matrix.toArray2



// ********************************
// ********************************
// Predicting from the larger set
// ********************************
// ********************************
//let generalModel = @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\Skin_NonSkin_adj.csv"
//
//let rawData = Common.readData generalModel
//let allFeatures = logisiticReg.createFeatureMatrix rawData 
//let allClassifiers = logisiticReg.createClassificationMatrix rawData 
//let firstFiftyThousandFeatures = logisiticReg.getSubSetOfMatrix allFeatures (10000, 20000)
//let firstfiftyThousandClassifiers = logisiticReg.getSubSetOfMatrix allClassifiers (10000, 20000)
//let allPredict = logisiticReg.predict firstFiftyThousandFeatures
//let fiftyThousandError = logisiticReg.calculateError allPredict firstfiftyThousandClassifiers
//
//let htTheta = logisiticReg.sigmoid(firstFiftyThousandFeatures*logisiticReg.finalTheta)


// ********************************
// ********************************
// Gradient descent
// ********************************
// ********************************
//let costs = logisiticReg.costs
//let grads = logisiticReg.gradients
//let init_theta = logisiticReg.initialTheta
//
//// descent function
//let firstDescent = logisiticReg.descent init_theta
//
//// descent function broken out
//let m      = (float logisiticReg.features.RowCount)
//let hx     = logisiticReg.sigmoid (logisiticReg.features*init_theta)
//let h      = hx-logisiticReg.classifications 
//let delt_J = logisiticReg.features 
//                |> Matrix.mapRows (fun i row -> h.[i, 0]*row)
//                |> Matrix.sumCols
//                |> Matrix.Build.DenseOfRowVectors
//((1.0/m) * delt_J.Transpose())

// ********************************
// ********************************
// Cost
// ********************************
// ********************************
//let theta = Matrix.Build.DenseOfColumnVectors(logisiticReg.gradients.Column(2))
//let m     = (float logisiticReg.X_train.RowCount) 
//let hx    = logisiticReg.sigmoid (logisiticReg.X_train*theta)
//
//let costs = logisiticReg.y_train
//                            |> Matrix.mapi (fun i j y_i -> match y_i with
//                                                            | 1.0 -> log(hx.[i, 0])
//                                                            | _   -> log(1.0-hx.[i, 0]))
//                            |> Matrix.sum

// ********************************
// ********************************
// New data
// ********************************
// ********************************
let path = @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\binary.csv"

let raw = Common.readData path
let lambda = 1.0
let alpha = 0.01 
let logisiticReg = LogisticRegression.LogReg(alpha, lambda, 100, 0.9, raw)


let indicies = logisiticReg.indices
let costs = logisiticReg.costs
let finalTheta = logisiticReg.finalTheta
let gradients = logisiticReg.gradients
logisiticReg.y_test
let pred = logisiticReg.sigmoid(logisiticReg.X_test*finalTheta)
let compare = pred-logisiticReg.y_test
let incorrentPredictions = compare 
                                        |> Matrix.toSeq 
                                        |> Seq.filter (fun x -> x <> 0.0)
                                        |> Seq.toList

let error = logisiticReg.error