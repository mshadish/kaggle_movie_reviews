create table movie_ratings (
phrase_id integer,
sentence_id integer,
phrase varchar(500),
sentiment integer
);

-- created a copy without the header
copy movie_ratings from '/Users/mshadish/kaggle/train2.tsv';

select count(1) from movie_ratings;

select * from movie_ratings;

select phrase, count(1) from movie_ratings
group by phrase
having count(1) = 1
order by count(1) desc;

select count(distinct sentence_id) from movie_ratings;

select sentiment, count(1) from movie_ratings group by sentiment order by sentiment;