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

// define topology - list of integers
// This would be a NN with..
// 1 input layer defined by the feature data
// 2 hidden layers having 6 nodes each
// an output layer with one class
let topology = [X_train.ColumnCount; 6; 6; 1;]

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
            let newAcc = List.append acc [(DenseMatrix.zero<float> topology.[count+1] (topology.[count]+1))]
            buildThetas newAcc (count+1)
        else
            acc

    buildThetas ([(DenseMatrix.zero<float> topology.[1] (topology.[0]))]) 1

    

let forwardPropagation (thetas: List<Matrix<float>>): List<Matrix<float>> = 
    let rec forward (acc: List<Matrix<float>>) (count:int) = 
        let layer = acc.[count]
        let tn    = thetas.[count]
        let zn    = tn*layer.Transpose()

        if count < (topology.Length-2) then
            let an    = addBasis (sigmoid (zn.Transpose()))
            let newAcc = List.append acc [an]
            forward newAcc (count+1)
        else
            let an    = (sigmoid (zn.Transpose()))
            List.append acc [an]

    forward [X_train] 0


// Recursive back propagation
let backPropagation (thetas: List<Matrix<float>>) (layers: List<Matrix<float>>) =
    let rec back (acc: List<Matrix<float>>) (countDown:int) =
        let tn  = thetas.[countDown] 
        let ln  = acc.[(acc.Length-1)]
        
        let an  = layers.[countDown]
        let gz = an.*(1.-an)
        
        printfn "%A" ln
        printfn "%A" tn
        
        let dn = (ln*tn).*gz
        
        printfn "%A" dn
        printfn "%i" countDown

        let newAcc = List.append acc [dn]

        if countDown = 0 then
            newAcc
        else 
            back newAcc (countDown-1)

    let firstError = layers.[(topology.Length-1)] - y_train
    back [firstError] (topology.Length-2)


// Call forwardPropagation
let layers = forwardPropagation initialTheta
let backpp = backPropagation initialTheta layers



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




