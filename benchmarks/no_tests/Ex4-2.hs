module Ex4_2 where

import Data.Vector

{-@ unsafeLookup :: <mask> @-}
unsafeLookup :: Int -> Vector a -> a
unsafeLookup index vec = vec ! index