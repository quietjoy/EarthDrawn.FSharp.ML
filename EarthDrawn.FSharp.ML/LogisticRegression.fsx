#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r @"..\packages\MathNet.Numerics.FSharp.3.10.0\lib\net40\MathNet.Numerics.FSharp.dll"

open System
open FSharp.Data
open MathNet.Numerics.LinearAlgebra

// ********************************    
// BUILDING MATRICIES
// ********************************
//let M = Matrix<float>.Build
//let V = Vector<float>.Build

// Turn the array of string[] to an array of float[]
// TODO: Take into account cultural differences of floats
let cast2float (x : string []) = 
    x |> Seq.map (fun s -> float s)

// Read in csv and return a Matrix
let readData (path : string) = 
    let csv = CsvFile.Load(path)
    let f2d = csv.Rows 
                |> Seq.map (fun x -> x.Columns) 
                |> Seq.map (fun x -> cast2float x)
                |> array2D
    matrix f2d


// ********************************
// MAIN LOGISTIC REGRESSION ALGORITHM
// ********************************

let sigmoid z = 1.0 / (1.0 + exp -z)

let cost (y:Matrix) (X:Matrix<float>) (theta:Matrix<float>) = 
    let h = (X.Transpose()*theta) |> Matrix.map(fun x -> log(sigmoid x))
    let firstTerm = -y*h 
    let secondTern = (1-y)*h
    (1/(float X.RowCount))*firstTern*secondTerm |> Matrix.

//let logReg (X:Matrix<float>) (y:Matrix<float>)=
//    (X.Transpose()*X).Inverse()*X.Transpose()*y

// ********************************
// VARIABLES
// ********************************
let data = 
    readData @"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\ex2data1.csv"

let X = M.Dense(data.RowCount, 1, 1.0).Append(data.RemoveColumn(data.ColumnCount-1))
let y = M.DenseOfColumnVectors(data.Column(data.ColumnCount-1))
let theta = M.Dense(X.RowCount, 1, 0.0)

let h = (X.Transpose()*theta).Enumerate() 
                |> Seq.map sigmoid
                
                
let firstTerm = -y*log(h)



let csv = CsvFile.Load(@"C:\Users\andre\Source\OSS\EarthDrawn.FSharp.ML\TestingData\LogisitcRegression\ex2data1.csv")
let f2d = csv.Rows 
            |> Seq.map (fun x -> x.Columns) 

let h = (X.Transpose()*theta).Enumerate() 
                |> Seq.map sigmoid

let rArray = f2d |> Seq.map(fun x -> string[] x) |> array2D