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
let topolgy = [6; 6; 1;]

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

let iterations = topolgy.Length

// sigmoid
let sigmoid (z: Matrix<float>): Matrix<float> = 
        z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))

// add basis term to matrix
let addBasis (zz: Matrix<float>) =
    Matrix.Build.Dense(zz.RowCount, 1, 1.0).Append(zz)

// first iteration - if count = 0
let t1 = DenseMatrix.zero<float> topolgy.[0] X_train.ColumnCount
let z1 = t1*X_train.Transpose()
let a2 = addBasis (sigmoid (z1.Transpose()))

// other iterations
let t2 = DenseMatrix.zero<float> topolgy.[1] (topolgy.[0]+1)
let z2 = t2*a2.Transpose()
let a3 = addBasis(sigmoid (z2.Transpose()))


// last iteration - if count = transpose.Length
let t3 = DenseMatrix.zero<float> topolgy.[2] (topolgy.[1]+1)
let z3 = t3*a3.Transpose()
let a4 = sigmoid (z3.Transpose())


//let rec feedForward (a: Matrix<float>) (count:int): Matrix<float> = 
//    if count = 0 then
//        let tn = DenseMatrix.zero<float> topolgy.[0] X_train.ColumnCount
//        let zn = tn*X_train.Transpose()
//        let an = addBasis (sigmoid (zn.Transpose()))
//        feedForward an (count+1)
//    else if count < topology.Length then
//        let tn = DenseMatrix.zero<float> topolgy.[(count+1)] (topolgy.[count]+1)
//        let zn = tn*a.Transpose()
//        let an = addBasis (sigmoid (zn.Transpose()))
//        feedForward an (count+1)
//    let tn = DenseMatrix.zero<float> topolgy.[(count+1)] (topolgy.[count]+1) 
//    sigmoid (tn*a.Transpose())

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



    
