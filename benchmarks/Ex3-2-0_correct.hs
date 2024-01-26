module Ex3_2_0 where

import Prelude hiding (abs)

{-@ abs :: Int -> Nat @-}
abs :: Int -> Int
abs n
    | 0 < n     = n
    | otherwise = 0 - n

{-@ test1 :: Nat @-}
test1 = abs (-7)
test1' = test1 == 7

{-@ test2 :: Nat @-}
test2 = abs 42
test2' = test2 == 42