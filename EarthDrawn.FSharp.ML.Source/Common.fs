module Common
    open FSharp.Data
    open System
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Double

    // ********************************    
    // BUILDING MATRICIES AND READING DATA
    // ********************************
    // Turn the array of string[] to an array of float[]
    let castToFloatList (x : string []): List<float> = 
        x |> Seq.map (fun s -> float s) |> Seq.toList

    let rng = new Random()
    let shuffle (arr : 'a array) =
        let array = Array.copy arr
        let n = array.Length
        for x in 1..n do
            let i = n-x
            let j = rng.Next(i+1)
            let tmp = array.[i]
            array.[i] <- array.[j]
            array.[j] <- tmp
        array |> Array.toList

    // Read in csv and return a shuffled Matrix of the data
    let readData (path : string): Matrix<float> = 
        let csv = CsvFile.Load(path)
        let f2d = csv.Rows 
                        |> Seq.map (fun x -> x.Columns) 
                        |> Seq.map (fun x -> castToFloatList x)
                        |> Seq.toArray
                        |> shuffle
        matrix f2d


    let getIndicies (size:int): (List<(int * int)>) =
        let trainIndex =  int (floor ((float size)*0.6))
        let cvIndex = trainIndex + (int (floor ((float size)*0.2)))
        [(0, trainIndex); (trainIndex, cvIndex); (cvIndex, size)]

    // Generate subset of matrix
    let getSubSetOfMatrix (data: Matrix<float>) (indicies: int * int): (Matrix<float>) =
        let rowCount = (snd indicies) - (fst indicies)
        data.SubMatrix((fst indicies), rowCount, 0, data.ColumnCount)


    // normalize a vector to have a mean of 0
    let normalize (x: Vector<float>): Vector<float> =
        let max  = x.Maximum()
        let min  = x.Minimum()
        let range = max - min
        let mean = (x |> Vector.sum) / (float x.Count)
        x |> Vector.map (fun x_i -> (x_i-mean)/range)

    // create feature matrix
    let createFeatureMatrix (data:Matrix<float>) (norm: Boolean) = 
        if norm then
            let unNormalizedFeatures = Matrix.Build.Dense(data.RowCount, 1, 1.0)
                                        .Append(data.RemoveColumn(data.ColumnCount-1))
            unNormalizedFeatures |> Matrix.mapCols (fun i x -> if (i <> 0) then normalize x else x)
        else
            Matrix.Build.Dense(data.RowCount, 1, 1.0)
                            .Append(data.RemoveColumn(data.ColumnCount-1))

    // create classification matrix
    let createClassificationMatrix (data:Matrix<float>) =
        Matrix.Build.DenseOfColumnVectors(data.Column(data.ColumnCount-1))