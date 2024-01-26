module Ex6_3 where

import Prelude hiding (head)
import Data.Maybe

{-@ die :: {v:String | false} -> a @-}
die msg = error msg

{-@ measure notEmpty @-}
notEmpty :: [a] -> Bool
notEmpty [] = False
notEmpty (_:_) = True

{-@ type NEList a = {v:[a] | notEmpty v} @-}

{-@ head :: NEList a -> a @-}
head (x:_) = x
head [] = die "Fear not! 'twill ne'er come to pass"

{-@ safeHead :: <mask> @-}
safeHead :: [a] -> Maybe a
safeHead xs
    | Prelude.null xs = Nothing
    | otherwise = Just $ head xs

{-@ null :: [a] -> Bool @-}
null [] = True
null (_:_) = False