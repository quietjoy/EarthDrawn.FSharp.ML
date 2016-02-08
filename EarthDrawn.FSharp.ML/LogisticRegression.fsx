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
let logisiticReg = LogisticRegression.LogReg(path, alpha, lambda, 100, raw)


let costs = logisiticReg.costs |> Seq.toArray

let indicies = logisiticReg.indices
let last = logisiticReg.features
let x_train = logisiticReg.X_train
let y_train = logisiticReg.y_train
let x_cv = logisiticReg.X_cv

let fT = logisiticReg.finalTheta
let test = matrix [[1.0; 170.0; 190.0; 247.0]]
let r = test*fT

// TODO
// Implement Gradient descent
// Add Regularization

// ********************************    
// BUILDING MATRICIES
// ********************************

// Turn the array of string[] to an array of float[]
// TODO: Take into account cultural differences of floats
let castToFloatList (x : string []) = 
    x |> Seq.map (fun s -> float s) |> Seq.toList

// Read in csv and return a Matrix
let readData (path : string) = 
    let csv = CsvFile.Load(path)
    let f2d = csv.Rows 
                |> Seq.map (fun x -> x.Columns) 
                |> Seq.map (fun x -> castToFloatList x)
                |> Seq.toList
    matrix f2d


// ********************************
// MAIN LOGISTIC REGRESSION ALGORITHM
// ********************************

// Compute the sigmoid function
let sigmoid z = 1.0 / (1.0 + exp -z)

// Calculate the regularized cost associated with a the proposed weights (theta)
let computeCost (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (lambda: float) = 
    let m = (float X.RowCount)
    let h = (X*theta) |> Matrix.map(fun x -> log(sigmoid x))
    let h1 = (X*theta) |> Matrix.map(fun x -> log(1.0-sigmoid x)) 
    let sum = (-y.*h) - ((y |> Matrix.map (fun x -> x-1.0)).*h1) |> Matrix.sum
    let J = (1.0/(float X.RowCount))*sum
    J + (lambda/(2.0*m))*(theta |> Matrix.sum)

// One step of gradient descent
let descent (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (alpha: float) =
    let m = (float X.RowCount)
    let hx = (X*theta) |> Matrix.map(fun x -> sigmoid x)
    let h = hx-y
    let sum = X |> Matrix.mapRows (fun i row -> h.[i, 0]*row) |> Matrix.sumCols 
    Matrix.Build.DenseOfColumnVectors (alpha*(1.0/m)*sum) 
//    (alpha*(1.0/m)*sum)

// Tail recursive gradient descent
//let rec gradientDescent (count: int) (gradAccum:Matrix<float>) = 
//    if count = 0 then
//        gradientDescent (count+1) (gradAccum.Append(descent y X theta alpha)) 
//    elif count <= X.RowCount then
//        let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count-1))
//        gradientDescent (count+1) (gradAccum.Append(descent y X prevTheta alpha))       
//    else 
//        gradAccum


// Not sure how this function is used right now
let costFunction (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (alpha: float) = 
    let t1f = computeCost y X theta alpha
    let t2f = descent y X theta alpha
    (t1f, t2f)
        
// ********************************
// VARIABLES
// ********************************
let data = 
    readData @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\ex2data1.csv"

let X = Matrix.Build.Dense(data.RowCount, 1, 1.0)
                .Append(data.RemoveColumn(data.ColumnCount-1))
let y = Matrix.Build.DenseOfColumnVectors(data.Column(data.ColumnCount-1))
let theta = Matrix.Build.Dense(X.ColumnCount, 1, 0.0)

let gg = [|0.0, 0.0, 0.0|]


let iterations = 100

// let g, theta_final = costFunction y X theta alpha


 