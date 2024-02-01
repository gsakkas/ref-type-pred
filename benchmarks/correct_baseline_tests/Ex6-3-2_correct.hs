module Ex6_3_2 where

import Prelude hiding (tail)
import Data.Maybe

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty [] = False
notEmpty (_:_) = True

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ tail :: NEList a -> [a] @-}
tail (_:xs) = xs
tail []     = die "Relaxeth! this too shall ne'er be"

test1 = tail [1, 3, 2, 42] == [3, 2, 42]

-- emptyList :: [Int]
-- emptyList = []
-- test2 = tail emptyList == undefined