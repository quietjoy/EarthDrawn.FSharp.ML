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
let sigmoid z = 1.0 / (1.0 + exp -z)

let computeCost (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (lambda: float) = 
    let m = (float X.RowCount)
    let h = (X*theta) |> Matrix.map(fun x -> log(sigmoid x))
    let h1 = (X*theta) |> Matrix.map(fun x -> log(1.0-sigmoid x)) 
    let sum = ((-y.*h) - ((y |> Matrix.map (fun x -> x-1.0)).*h1) |> Matrix.sum)
    let J = (1.0/(float X.RowCount))*sum
    J + (lambda/(2.0*m))*(theta |> Matrix.sum)

let descent (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (alpha: float) =
    let m = (float X.RowCount)
    let hx = (X*theta) |> Matrix.map(fun x -> sigmoid x)
    let h = hx-y
    let sum = X |> Matrix.mapRows (fun i row -> h.[i, 0]*row) |> Matrix.sumCols 
    Matrix.Build.DenseOfColumnVectors (alpha*(1.0/m)*sum) 
//    (alpha*(1.0/m)*sum)

 
let gradientDescent (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>)
                    (lambda: float)   (alpha: float)    (iterations: int) =         
    let mutable returnMatrix = Matrix.Build.Dense(1, 3, 0.0)
    
    let place (n: int) = 
        match n with 
        | 0 -> returnMatrix.Add(descent y X theta alpha)
        | _ -> returnMatrix.Add(descent y X (Matrix.Build.DenseOfColumnVectors (returnMatrix.Row((n-1)))) alpha)
    
    let placeHolder = Array.init iterations (fun n -> n + 1) |> Array.map (fun i -> place i)
    
    returnMatrix

//    let i = (Array2D.zeroCreate<float> iterations 3)
//    let init_gradient =  Matrix.Build.DenseOfArray i
//    init_gradient |> Matrix.(fun t -> Some(t, descent y X t alpha))

let costFunction (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (alpha: float) = 
    let t1f = computeCost y X theta alpha
    let t2f = descent y X theta alpha
    (t1f, t2f)

let rec fixedPoint f x =
    let f_x = f x
    if f_x = x then x else fixedPoint f f_x;
        
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

let mutable returnMatrix = Matrix.Build.Dense(3, 1, 0.0)
//let oo = (descent y X theta alpha)
//let p = (descent y X theta alpha)
//returnMatrix.Append(oo)
//returnMatrix.Append(p)

let place (n: int) = 
    match n with 
    | 0 -> returnMatrix.Append(descent y X theta alpha)
    | _ -> returnMatrix.Append(descent y X (Matrix.Build.DenseOfRowVectors (returnMatrix.Column((n-1)))) alpha)
    
let placeHolder = Array.init iterations (fun n -> n + 1) |> Array.map (fun i -> place i)
    
returnMatrix
