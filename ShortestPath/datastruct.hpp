#pragma once

#define LL long long

typedef struct _edge{
    LL from, to;
    bool operator < (const struct _edge& e) const {
        if(from == e.from)
            return to < e.to;
        return from < e.from;
    }
} _edge;

class _queue{
#define MAXQUE 1000000
    LL queue[MAXQUE];
    LL queLen, queHead, queTail;
    bool check() const;
public:
    _queue();
    void push(LL);
    LL front() const;
    LL pop();
    void clear();
    bool empty();
};

_queue::_queue(){
    queLen = 0;
    queHead = queTail = 0;
}

bool _queue::check() const {
    if(queLen >= MAXQUE)
        return false;
    return true;
}

void _queue::push(LL v){
    if(!check()){
        return;
    }
    queue[queTail++] = v;
    queLen++;
    if(queTail >= MAXQUE)
        queTail -= MAXQUE;
}

LL _queue::front() const {
    return queue[queHead];
}

LL _queue::pop(){
    if(queLen <= 0){
        return -1;
    }
    LL v = queue[queHead];
    queHead++;
    queLen--;
    if(queHead >= MAXQUE)
        queHead -= MAXQUE;
    return v;
}

bool _queue::empty(){
    if(queLen <= 0)
        return true;
    return false;
}