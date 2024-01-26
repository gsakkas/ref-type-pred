module Ex6_3_1 where

import Prelude hiding (head)
import Data.Maybe

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty [] = False
notEmpty (_:_) = True

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ head :: <mask> @-}
head :: [a] -> a
head (x:_) = x
head []    = die "Fear not! 'twill ne'er come to pass"