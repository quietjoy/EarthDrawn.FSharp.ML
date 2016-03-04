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

let λ = 1.0
let α = 0.01
let threshold = 0.5
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
let dist = MathNet.Numerics.Distributions.ContinuousUniform(-0.999999, 0.999999)
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

let predictByThreshold (p : Matrix<float>): Matrix<float> = 
    p |> Matrix.map (fun x -> if x >= threshold then 1.0 else 0.0)

let countOnetoZeroRatio (p: Matrix<float>): float =
    let count = p |> Matrix.toSeq |> Seq.countBy (fun x -> x = 0.0) |> Seq.toArray
    let zeros = float (snd (count.[1]))
    let ones  = float (snd (count.[0]))
    ones / zeros

let explain (p: Matrix<float>): float = 
    countOnetoZeroRatio (predictByThreshold p)

// Recursive forward propogation
let forwardPropagation (thetas: List<Matrix<float>>): List<Matrix<float>> = 
    let c = thetas.Length-1
    let rec propagation (acc: List<Matrix<float>>) (count:int) = 
        let an = acc.[count]
        let tn = thetas.[count]
        let zn = an*tn.Transpose()
        let hx = sigmoid zn

        if count < c then
            propagation (List.append acc [(addBasis hx)]) (count+1)
        else
            List.append acc [hx]

    propagation [X_train] 0



// Recursive back propagation
let backPropagation (thetas: List<Matrix<float>>) (layers: List<Matrix<float>>) =
    let rec propagation (acc: List<Matrix<float>>) (countDown:int) =
        let tn  = thetas.[countDown]
        let ln  = acc.[(acc.Length-1)]
        let an  = layers.[countDown]
        let gz  = sigmoidGradient(an)

        let dn = removeFirstColumn ((ln*tn).*gz)

        // there is no delt(1) term
        if countDown = 1 then
            List.append acc [dn]
        else 
            propagation (List.append acc [dn]) (countDown-1)
    
    // calculate first error
    let fe = layers.[(layers.Length-1)] - y_train

    // recursively apply back propogation
    let bp = propagation [fe] (thetas.Length-1)

    // return the list in reverse order
    bp |> List.rev

// Taken from LogisticRegression.fs - Calculate the error
let calculateError (predictions:Matrix<float>) (y: Matrix<float>): float =
        let compare = predictions-y
        let incorrentPredictions = compare 
                                    |> Matrix.toSeq 
                                    |> Seq.filter (fun x -> x <> 0.0)
                                    |> Seq.toList
        ((float incorrentPredictions.Length) / (float y.RowCount))

let initFunc (m: Matrix<float>) =
    m |> Matrix.map (fun x -> x)

let removeBasisAndUnroll (m: Matrix<float>): Matrix<float> = 
    let noBasis = removeFirstColumn m
    let rowCount = noBasis.RowCount*noBasis.ColumnCount
    let indexNoBasis (i: int): float =
        let col = int (Math.Floor((float i)/ float noBasis.ColumnCount)) 
        let row = i - (col * noBasis.ColumnCount)
        printfn ("%i %i") row col
        noBasis.[row, col]

    Matrix.Build.Dense(rowCount, 1, (fun i j -> indexNoBasis i))

// cost function with feature normalization
let recursiveRegularizedCostFunction (hx:Matrix<float>) (thetas:List<Matrix<float>>): float =
    
    let m     = (float X_train.RowCount) 
    let sum   = y_train
                |> Matrix.mapi (fun i j y_i -> match y_i with
                                                | 1.0 -> log(hx.[i, 0])
                                                | _   -> log(1.0-hx.[i, 0]))
                |> Matrix.sum
    
    let regTerm = thetas 
                    |> List.map (fun m -> m
                                        |> removeBasisAndUnroll
                                        |> Matrix.map (fun y_i -> y_i**2.0)
                                        |> Matrix.sum)
                    |> List.sum

    (-1.0/m*sum) + (λ/(2.0*m)*(regTerm))


let accumDelta (forwardLayers:List<Matrix<float>>) (backwardsLayers:List<Matrix<float>>): List<Matrix<float>> =
    let c = forwardLayers.Length-1
    let rec getDeltas (accum:List<Matrix<float>>) (count: int) =
        if count < c then
            let d = backwardsLayers.[count].Transpose() * forwardLayers.[count]
            getDeltas (List.append accum [d]) (count+1)
        else
            accum
    getDeltas [] 0

let findPartialDerivates (thetas:List<Matrix<float>>) (deltAccum:List<Matrix<float>>): List<Matrix<float>> =
    let m = float deltAccum.[0].RowCount
    let c = thetas.Length
    let rec calculate (accum:List<Matrix<float>>) (count: int) =
        if count < c then
            let d = (1.0/m)*deltAccum.[count] + λ*thetas.[count]
            calculate (List.append accum [d]) (count+1)
        else 
            accum
    calculate [(1.0/m)*deltAccum.[0]] 1

// calculate gradient
let calculateGradient (partialDerivatives: List<Matrix<float>>) (thetas: List<Matrix<float>>): List<Matrix<float>> =
    let c = thetas.Length
    let rec updateThetas (accum: List<Matrix<float>>)(count: int) : List<Matrix<float>> =
        if count < c then
            let ct = thetas.[count]
            let pd = partialDerivatives.[count]
            let ut = ct-(pd*α)
            updateThetas (List.append accum [ut]) (count+1)
        else 
            accum
    updateThetas [] 0

// recursively applies descent function
let gradientDescent (thetas:List<Matrix<float>>) (iterations: int): List<List<Matrix<float>>> =
    let rec descent (count: int) (accum: List<List<Matrix<float>>>): List<List<Matrix<float>>> =
        if count < iterations then
            let ct = accum.[(count-1)]
            let fp = forwardPropagation ct
            let bp = backPropagation ct fp
            let da = accumDelta fp bp
            let pd = findPartialDerivates ct da
            let gd = calculateGradient pd ct
            descent (count+1) (List.append accum [gd])
        else 
            accum
    descent 1 [thetas]
            
let NNRun (finalThetas: List<Matrix<float>>): Matrix<float> =
    let rec run (curr: Matrix<float>) (count: int): Matrix<float> =
        if count < finalThetas.Length then
            let theta = finalThetas.[(count-1)]
            run ((curr*theta.Transpose()).Transpose()) (count+1)
        else 
            curr
    let tf = finalThetas.[0]
    run (X_train*tf.Transpose()) 1

// Call forwardPropagation
let forwardLayers = forwardPropagation initialTheta
// Call backpropogation
let backwardLayers = backPropagation initialTheta forwardLayers
// accumulate Deltas
let deltAccum = accumDelta forwardLayers backwardLayers
// find partial derivatives
let partialDerivatives = findPartialDerivates initialTheta deltAccum
// one step of gradient descent
let updatedThetas = calculateGradient partialDerivatives initialTheta
// gradient descent 100 iterations
let gThetas = gradientDescent partialDerivatives 100
// get the final thetas
let finalThetas = gThetas.[(gThetas.Length-1)]
// make predicitions
let predicitions = NNRun finalThetas

//let f = sigmoid(X_train*(finalThetas.[0]).Transpose())
//let s = sigmoid(f*finalThetas.[1])
//let t = sigmoid(s*finalThetas.[2].Transpose())

let f = sigmoid(X_train*(initialTheta.[0]).Transpose())
let s = sigmoid(f*initialTheta.[1])
let t = sigmoid(s*initialTheta.[2].Transpose())

//
////calculate error
//let asd = calculateError p y_train
//
//
//let p = 
//let count = p |> Matrix.toSeq |> Seq.countBy (fun x -> x = 0.0) |> Seq.toArray
//let zeros = float (snd (count.[1]))
//let ones  = float (snd (count.[0]))
//ones / zeros
// 

