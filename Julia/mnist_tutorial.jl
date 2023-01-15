using VegaLite, DataFrames, Query, VegaDatasets

cars = dataset("cars")

cars |> @select(:Acceleration, :Name) |> collect

function foo(data, origin)
    df = data |> @filter(_.Origin==origin) |> DataFrame
    return df |> @vlplot(:point, :Acceleration, :Horsepower)
end

p = foo(cars, "USA")
# p |> save("foo.png")

# so far, I have made very simple dataset process.
