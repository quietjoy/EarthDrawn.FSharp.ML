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

// load data 
let path = @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\Skin_NonSkinSample.csv"

let rawData = Common.readData path

// define topology - list of integers - first layer is defined by data 
// This would be a NN with 1 hidden layer having 6 nodes and
// an output layer with one class
let topology = [6; 6; 1;]

let λ = 0.01

let normalize = false

// features
let features = Common.createFeatureMatrix rawData false
// classifications
let classifications = Common.createClassificationMatrix rawData

// list of tuples (int * int) that are the indicies of the 
// train, cross validation and testing data
let indices = Common.getIndicies rawData.RowCount 

// Build training data
let X_train = Common.getSubSetOfMatrix features (indices.[0])
let y_train = Common.getSubSetOfMatrix classifications (indices.[0])

let iterations = topology.Length

// sigmoid
let sigmoid (z: Matrix<float>): Matrix<float> = 
        z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))

// add basis term to matrix
let addBasis (zz: Matrix<float>) =
    Matrix.Build.Dense(zz.RowCount, 1, 1.0).Append(zz)

let initialTheta: List<Matrix<float>> = 
    let max = (topology.Length-2)

    let rec buildThetas (acc:List<Matrix<float>>) (count:int) = 
        if count <= max then
            printfn "%i" count
            let newAcc = List.append acc [(DenseMatrix.zero<float> topology.[count+1] (topology.[count]+1))]
            buildThetas newAcc (count+1)
        else
            acc

    buildThetas ([(DenseMatrix.zero<float> topology.[0] X_train.ColumnCount)]) 1

    

let feedForward (thetas: List<Matrix<float>>): List<Matrix<float>> = 
    let rec forward (acc: List<Matrix<float>>) (count:int) = 
        printfn "%i" count
        let layer = acc.[count]
        let tn    = thetas.[count]
        let zn    = tn*layer.Transpose()
        let an    = addBasis (sigmoid (zn.Transpose()))

        if count < (topology.Length-2) then
            let newAcc = List.append acc [an]
            forward newAcc (count+1)
        else
            List.append acc [an]

    forward [X_train] 0

feedForward initialTheta

// cost function with feature normalization
let calculateCost (thetaV:Vector<float>): List<float> =
    let m     = (float X_train.RowCount) 
    let theta = Matrix.Build.DenseOfColumnVectors(thetaV)
    let hx    = sigmoid (X_train*theta)
    let sum   = y_train
                    |> Matrix.mapi (fun i j y_i -> match y_i with
                                                    | 1.0 -> log(hx.[i, 0])
                                                    | _   -> log(1.0-hx.[i, 0]))
                    |> Matrix.sum
    let regTerm = theta 
                    |> Matrix.mapi(fun i j y_i -> if (i<>0) then (y_i**2.0) else 0.0) 
                    |> Matrix.sum
    [(-1.0/m*sum) + (λ/(2.0*m)*(regTerm))]




