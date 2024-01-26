module Ex6_1_0 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ type Nat     = {v:Int | 0 <= v} @-}
{-@ type Pos     = {v:Int | 0 < v}  @-}
{-@ type NonZero = {v:Int | v /= 0} @-}

{-@ divide :: <mask> @-}
divide _ 0 = die "divide-by-zero"
divide x n = x `div` n