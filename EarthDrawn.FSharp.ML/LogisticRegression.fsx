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

let computeGradient (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) (alpha: float) =
    let m = (float X.RowCount)
    let hx = (X*theta) |> Matrix.map(fun x -> sigmoid x)
    let h = hx-y
    let sum = X |> Matrix.mapRows (fun i row -> h.[i, 0]*row) |> Matrix.sumCols 
    alpha*(1.0/m)*sum

let costFunction (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>) = 
    let t1f = computeCost y X theta
    let t2f = computeGradient y X theta
    (t1f, t2f)

let performGradientDescent (y:Matrix<float>) (X:Matrix<float>) (theta:Matrix<float>)
                           (lambda: float) (alpha: float) (iterations: int) =
    let gradient = 
    let J = [| for i in 1 .. iterations -> computeCost y X theta lamda |]
        
    

// ********************************
// VARIABLES
// ********************************
let data = 
    readData @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\ex2data1.csv"

let X = Matrix.Build.Dense(data.RowCount, 1, 1.0)
                .Append(data.RemoveColumn(data.ColumnCount-1))
let y = Matrix.Build.DenseOfColumnVectors(data.Column(data.ColumnCount-1))
let theta = Matrix.Build.Dense(X.ColumnCount, 1, 0.0)
let δ = System.Double.Epsilon ** (1.0 / 3.0)

let lamda = 1.0
let g, theta_final = costFunction y X theta


//let bfgsMin (initialGuess: float[]) costFunction gradientFunction =
//    let lbfgsb = DotNumerics.Optimization.L_BFGS_B()
//    lbfgsb.ComputeMin(costFunction, gradientFunction, initialGuess);

let J = [| for i in 1 .. 100 -> computeCost y X theta lamda |]