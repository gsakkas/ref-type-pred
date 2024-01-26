module Ex4_2 where

import Data.Vector

{-@ unsafeLookup :: <mask> @-}
unsafeLookup :: Int -> Vector a -> a
unsafeLookup index vec = vec ! index

test1 = unsafeLookup 0 (fromList [1, -2, 3]) == 1
-- test2 = unsafeLookup (-3) (fromList [1, -2, 3, 42]) == undefined
-- test3 = unsafeLookup 7 (fromList [1, 3, 42]) == undefined