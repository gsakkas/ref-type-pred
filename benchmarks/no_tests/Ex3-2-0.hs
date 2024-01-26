module Ex3_2_0 where

import Prelude hiding (abs)

{-@ abs :: <mask> @-}
abs :: Int -> Int
abs n
    | 0 < n     = n
    | otherwise = 0 - n