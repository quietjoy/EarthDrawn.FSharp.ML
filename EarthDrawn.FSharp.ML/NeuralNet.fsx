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

// sigmoid function
let sigmoid (z: Matrix<float>): Matrix<float> = 
        z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))

// sigmoid gradient funciton
let sigmoidGradient (z: Matrix<float>): Matrix<float> = 
    let sigZ = sigmoid z
    sigZ.*(1.0 - sigZ)

// add basis term to matrix
let addBasis (zz: Matrix<float>) =
    Matrix.Build.Dense(zz.RowCount, 1, 1.0).Append(zz)

// remove basis term
let removeFirstColumn (zz: Matrix<float>) =
    let rowCount = zz.RowCount
    zz.SubMatrix(0, rowCount, 1, (zz.ColumnCount-1)) 

// remove first row
let removeFirstRow (zz: Matrix<float>) =
    let columnCount = zz.ColumnCount
    zz.SubMatrix(1, (zz.RowCount-1), 0, columnCount) 

// Unroll a matrix into a row matrix
let unroll (zz: Matrix<float>) =
    let cc = zz.ColumnCount
    let rc = zz.RowCount
    Seq.init (rc*cc) (fun r -> ((float (r/cc)), (float ((rc-r)/rc))    ))  |> Seq.toArray
//    DenseMatrix.ofColumnSeq x

// Build a random matrix from continuous uniform distribution
let dist = MathNet.Numerics.Distributions.ContinuousUniform(0.0, 0.999999)
let getRandom (row: int) (col: int): Matrix<float> = 
    Matrix<float>.Build.Random(row, col, dist)
    
// Build a list of matricies that represent the thetas
// basis term not add for first theta b/c 
// X_train accurately represents the size with basis term
let initialTheta: List<Matrix<float>> = 
    let max = (topology.Length-2)

    let rec buildThetas (acc:List<Matrix<float>>) (count:int) = 
        if count <= max then
            let newAcc = List.append acc [(getRandom topology.[count+1] (topology.[count]+1))]
            buildThetas newAcc (count+1)
        else
            acc
    buildThetas ([(getRandom topology.[1] (topology.[0]))]) 1

let initialDeltAccums: List<Matrix<float>> = 
    let length = X_train.RowCount
    let t = Matrix.Build.Dense(length, 1, 0.0)
    let accum = topology |> List.mapi (fun i x -> 
                                    if i = 0 then Matrix.Build.Dense(length, (x-1), 0.0)
                                    else  Matrix.Build.Dense(length, x, 0.0))
    accum |> List.take (accum.Length - 1)


// Recursive forward propogation
let forwardPropagation (thetas: List<Matrix<float>>): List<Matrix<float>> = 
    let rec forward (acc: List<Matrix<float>>) (count:int) = 
        let layer = acc.[count]
        let tn    = thetas.[count]
        let zn    = tn*layer.Transpose()

        if count < (topology.Length-2) then
            let an     = addBasis (sigmoid (zn.Transpose()))
            let newAcc = List.append acc [an]
            forward newAcc (count+1)
        else
            let an = (sigmoid (zn.Transpose()))
            List.append acc [an]

    forward [X_train] 0


// Recursive back propagation
let backPropagation (thetas: List<Matrix<float>>) (layers: List<Matrix<float>>) (delts: List<Matrix<float>>) =
    let rec back (acc: List<Matrix<float>>) (countDown:int) =
        let tn  = thetas.[countDown]
        let ln  = acc.[(acc.Length-1)]
        let an  = layers.[countDown]
        let gz  = sigmoidGradient(an)
        let d   = delts.[countDown]

        let dn = d + removeFirstColumn ((ln*tn).*gz)
        let newAcc = List.append acc [dn]

        // don't calculate delta for a(1)
        if countDown = 1 then
            newAcc
        else 
            back newAcc (countDown-1)

    let firstError = (layers.[(topology.Length-1)] - y_train)
    back [firstError] (topology.Length-2)

// cost function with feature normalization
// Start here
let calculateCost (hx:Matrix<float>) (thetas:List<Matrix<float>>): List<float> =
    let m     = (float X_train.RowCount) 
    let sum   = y_train
                |> Matrix.mapi (fun i j y_i -> match y_i with
                                                | 1.0 -> log(hx.[i, 0])
                                                | _   -> log(1.0-hx.[i, 0]))
                |> Matrix.sum
    let regTerm = thetas 
                    |> List.map (fun m -> m
                                        |> Matrix.mapi(fun i j y_i -> if (i<>0) then (y_i**2.0) else 0.0)
                                        |> Matrix.sum)
                    |> List.sum

    [(-1.0/m*sum) + (λ/(2.0*m)*(regTerm))]

// Taken from LogisticRegression.fs
let calculateError (predictions:Matrix<float>) (y: Matrix<float>): float =
        let compare = predictions-y
        let incorrentPredictions = compare 
                                    |> Matrix.toSeq 
                                    |> Seq.filter (fun x -> x <> 0.0)
                                    |> Seq.toList
        ((float incorrentPredictions.Length) / (float y.RowCount))

let accumDelta (forwardLayers:List<Matrix<float>>) (backwardsLayers:List<Matrix<float>>): List<Matrix<float>> =
    let c = forwardLayers.Length
    let rec getDeltas (accum:List<Matrix<float>>) (count: int) =
        if count < c then
            let d = backwardsLayers.[count].Transpose() * forwardLayers.[count]
            getDeltas (List.append accum [d]) (count+1)
        else
            accum
    getDeltas [] 0

let findPartialDerivates (thetas:List<Matrix<float>>) (deltAccum:List<Matrix<float>>) (λ:float): List<Matrix<float>> =
    let m = float deltAccum.[0].RowCount
    let c = thetas.Length
    let rec calculate (accum:List<Matrix<float>>) (count: int) =
        if count < c then
            let d = (1.0/m)*deltAccum.[count] + λ*thetas.[count]
            calculate (List.append accum [d]) (count+1)
        else 
            accum
    calculate [(1.0/m)*deltAccum.[0]] 1

// Call forwardPropagation
let forwardLayers = forwardPropagation initialTheta
// Call backpropogation
let backwardLayers = (backPropagation initialTheta forwardLayers initialDeltAccums) |> List.rev


// partial derivatives

let deltAccum1 = backwardLayers.[0].Transpose() * forwardLayers.[0]
let deltAccum2 = backwardLayers.[1].Transpose() * forwardLayers.[1]
let deltAccum3 = backwardLayers.[2].Transpose() * forwardLayers.[2]

let deltAccum = accumDelta forwardLayers backwardLayers
let updatedThetas = findPartialDerivates initialTheta deltAccum 0.1 
