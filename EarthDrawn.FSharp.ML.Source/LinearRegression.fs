namespace EarthDrawn.FSharp.ML.Source

module LinearRegression =  
    open FSharp.Data
    open MathNet.Numerics.LinearAlgebra;
    open MathNet.Numerics.LinearAlgebra.Double;

    let M = Matrix<float>.Build
    let V = Vector<float>.Build

    // ********************************    
    // BUILDING MATRICIES AND READING DATA
    // ********************************
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

    let predict (data:Matrix<float>) (theta:Matrix<float>) =
        data*theta

    let calculateStandardError (p:Matrix<float>) (y:Matrix<float>) =
        let sum = (p-y).Enumerate() |> Seq.map (fun x -> x**2.0) |> Seq.sum
        sqrt (sum/(float (y.RowCount-2)))


    type LinearRegression(pa:string) = 
        member this.data = readData pa
        member this.X = M.Dense(this.data.RowCount, 1, 1.0)
                            .Append(this.data.RemoveColumn(this.data.ColumnCount-1))
        member this.y = M.DenseOfColumnVectors(this.data.Column(this.data.ColumnCount-1))
        member this.theta = linReg this.X this.y
        member this.modelPredictions = predict  this.X  this.theta
        member this.standardError = calculateStandardError this.p this.y

