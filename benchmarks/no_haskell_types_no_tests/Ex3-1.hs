module Ex3_1 where

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ type Zero    = {v:Int | v == 0} @-}
{-@ type NonZero = {v:Int | v /= 0} @-}

{-@ divide :: Int -> NonZero -> Int @-}
divide :: Int -> Int -> Int
divide _ 0 = die "divide by zero"
divide n d = n `div` d

{-@ avg :: <mask> @-}
avg xs = divide total n
    where
        total = sum xs
        n = length xs