#r @"..\packages\FSharp.Data.2.2.5\lib\net40\FSharp.Data.dll"
#r @"..\packages\MathNet.Numerics.3.10.0\lib\net40\MathNet.Numerics.dll"

open FSharp.Data
open MathNet.Numerics.LinearAlgebra;
open MathNet.Numerics.LinearAlgebra.Double;


// ********************************    
// BUILDING MATRICIES
// ********************************
let M = Matrix<float>.Build
let V = Vector<float>.Build

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
    M.DenseOfArray f2d

// ********************************
// MAIN LINEAR REGRESSION ALGORITHM
// ********************************
let linReg (X:Matrix<float>) (y:Matrix<float>)=
    (X.Transpose()*X).Inverse()*X.Transpose()*y

// ********************************
// POTENTIAL TYPE EXTENTIONS
// ********************************
let predict (data:Matrix<float>) (theta:Matrix<float>) =
    data*theta

//let standardError (p:Matrix<float>) (y:Matrix<float>) =
//    ((p-y).*(p-y)).RowSums

// ********************************
// VARIABLES
// ********************************
let data = 
    readData @"C:\Users\andre\Documents\WebCourses\MachineLearning\Week2\machine-learning-ex1\ex1\ex1data1_headers.csv"

let X = M.Dense(data.RowCount, 1, 1.0).Append(data.RemoveColumn(data.ColumnCount-1))
let y = M.DenseOfColumnVectors(data.Column(data.ColumnCount-1))
let theta = linReg X y
let p = predict  X  theta

//let s = standardError p y