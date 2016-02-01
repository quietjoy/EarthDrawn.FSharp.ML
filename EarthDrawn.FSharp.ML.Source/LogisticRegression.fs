namespace EarthDrawn.FSharp.ML.Source

module LogisticRegression =
    open System
    open FSharp.Data
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Double

    // ********************************    
    // BUILDING MATRICIES AND READING DATA
    // ********************************
    // Turn the array of string[] to an array of float[]
    let castToFloatList (x : string []): List<float> = 
        x |> Seq.map (fun s -> float s) |> Seq.toList

    // Read in csv and return a Matrix
    let readData (path : string): Matrix<float> = 
        let csv = CsvFile.Load(path)
        let f2d = csv.Rows 
                        |> Seq.map (fun x -> x.Columns) 
                        |> Seq.map (fun x -> castToFloatList x)
                        |> Seq.toList
        matrix f2d

    type LogReg (path: string, a0: float, l0: float, it: int) =
        member this.rawData = readData path
        member this.X = Matrix.Build.Dense(this.rawData.RowCount, 1, 1.0)
                                .Append(this.rawData.RemoveColumn(this.rawData.ColumnCount-1))
        member this.y = Matrix.Build.DenseOfColumnVectors(this.rawData.Column(this.rawData.ColumnCount-1))
        member this.alpha = a0
        member this.lambda = l0
        member this.initialTheta = Matrix.Build.Dense(this.X.ColumnCount, 1, 0.0)
        member this.iterations = it

        member this.sigmoid z = 1.0 / (1.0 + exp -z)

        member this.computeCost (theta:Matrix<float>) = 
            let m = (float this.X.RowCount)
            let h = (this.X*theta) |> Matrix.map(fun x -> log(this.sigmoid x))
            let h1 = (this.X*theta) |> Matrix.map(fun x -> log(1.0-(this.sigmoid x))) 
            let sum = (-this.y.*h) - ((this.y |> Matrix.map (fun x -> x-1.0)).*h1) |> Matrix.sum
            let J = (1.0/(float this.X.RowCount))*sum
            J + (this.lambda/(2.0*m))*(theta |> Matrix.sum)

        member this.descent (theta:Matrix<float>) =
            let m = (float this.X.RowCount)
            let hx = (this.X*theta) |> Matrix.map(fun x -> this.sigmoid x)
            let h = hx-this.y
            let sum = this.X |> Matrix.mapRows (fun i row -> h.[i, 0]*row) |> Matrix.sumCols 
            Matrix.Build.DenseOfColumnVectors (this.alpha*(1.0/m)*sum) 
        //    (alpha*(1.0/m)*sum)

        member this.gradientDescent (count: int) (gradAccum:Matrix<float>) = 
            if count = 0 then
                this.gradientDescent (count+1) (gradAccum.Append(this.descent this.initialTheta)) 
            elif count <= this.iterations then
                let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count-1))
                this.gradientDescent (count+1) (gradAccum.Append(this.descent prevTheta))       
            else 
                gradAccum

        member this.costFunction (theta:Matrix<float>) = 
            let t1f = this.computeCost theta
            let t2f = this.descent theta
            (t1f, t2f)