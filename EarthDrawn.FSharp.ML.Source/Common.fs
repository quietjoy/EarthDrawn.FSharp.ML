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
