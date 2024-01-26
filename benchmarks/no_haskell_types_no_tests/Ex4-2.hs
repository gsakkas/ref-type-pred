module Ex4_2 where

import Data.Vector

{-@ unsafeLookup :: <mask> @-}
unsafeLookup index vec = vec ! index