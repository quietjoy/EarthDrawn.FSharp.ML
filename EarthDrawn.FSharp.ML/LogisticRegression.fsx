#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r @"..\packages\MathNet.Numerics.3.10.0\lib\net40\MathNet.Numerics.dll"
#r @"..\packages\DotNumerics.1.1\lib\DotNumerics.dll"


#load "..\packages\MathNet.Numerics.FSharp.3.10.0\MathNet.Numerics.fsx"

open System
open FSharp.Data
open MathNet.Numerics.LinearAlgebra
open DotNumerics

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

// Perform gradient descent
//let gradientDescent (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>)
//                    (lambda: float)   (alpha: float)    (iterations: int) =         
//    let mutable returnMatrix = Matrix.Build.Dense(1, 3, 0.0)
//    let place (n: int) = 
//        match n with 
//        | 0 -> returnMatrix.Add(descent y X theta alpha)
//        | _ -> returnMatrix.Add(descent y X (Matrix.Build.DenseOfColumnVectors (returnMatrix.Row((n-1)))) alpha)
//    let placeHolder = Array.init iterations (fun n -> n + 1) |> Array.map (fun i -> place i)
//    returnMatrix


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

let lambda = 1.0
let alpha = 0.01 
let iterations = 100

let g, theta_final = costFunction y X theta alpha

let rr = gradientDescent X y theta lambda alpha iterations


// Testing a tail recursive gradient descent
let rm = Matrix.Build.Dense(3, X.RowCount, 0.0)

let rec recGradientDescent (count: int) (gradAccum:Matrix<float>) = 
    if count = 0 then
        recGradientDescent (count+1) (gradAccum.Append(descent y X theta alpha)) 
    elif count <= X.RowCount then
        let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count-1))
        recGradientDescent (count+1) (gradAccum.Append(descent y X prevTheta alpha))       
    else 
        gradAccum