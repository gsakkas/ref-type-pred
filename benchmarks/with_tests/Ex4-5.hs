module Ex4_5 where

import Prelude hiding (abs)
import Data.Vector
import qualified Data.Vector as V

{-@ LIQUID "--no-termination" @-}

{-@ absoluteSum :: V.Vector Int -> Nat @-}
absoluteSum     :: V.Vector Int -> Int
absoluteSum vec = go' 0 0
  where
    {-@ go' :: <mask> @-}
    go' :: Int -> Int -> Int
    go' acc i
      | i < sz    = go' (acc + abs (vec ! i)) (i + 1)
      | otherwise = acc
    sz            = V.length vec

{-@ abs :: Int -> Nat @-}
abs :: Int -> Int
abs x | x < 0     = 0 - x
      | otherwise = x

{-@ test1 :: Nat @-}
test1 = absoluteSum (fromList [1, -2, 3])
test1'= test1 == 6

{-@ test2 :: Nat @-}
test2 = absoluteSum (fromList [1, -2, 3, 42, 0])
test2' = test2 == 48