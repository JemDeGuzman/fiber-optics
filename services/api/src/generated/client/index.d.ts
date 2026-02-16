
/**
 * Client
**/

import * as runtime from './runtime/binary.js';
import $Types = runtime.Types // general types
import $Public = runtime.Types.Public
import $Utils = runtime.Types.Utils
import $Extensions = runtime.Types.Extensions
import $Result = runtime.Types.Result

export type PrismaPromise<T> = $Public.PrismaPromise<T>


/**
 * Model User
 * 
 */
export type User = $Result.DefaultSelection<Prisma.$UserPayload>
/**
 * Model DataBatch
 * 
 */
export type DataBatch = $Result.DefaultSelection<Prisma.$DataBatchPayload>
/**
 * Model DataSample
 * 
 */
export type DataSample = $Result.DefaultSelection<Prisma.$DataSamplePayload>
/**
 * Model ImageCapture
 * 
 */
export type ImageCapture = $Result.DefaultSelection<Prisma.$ImageCapturePayload>

/**
 * ##  Prisma Client ʲˢ
 *
 * Type-safe database client for TypeScript & Node.js
 * @example
 * ```
 * const prisma = new PrismaClient()
 * // Fetch zero or more Users
 * const users = await prisma.user.findMany()
 * ```
 *
 *
 * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
 */
export class PrismaClient<
  ClientOptions extends Prisma.PrismaClientOptions = Prisma.PrismaClientOptions,
  const U = 'log' extends keyof ClientOptions ? ClientOptions['log'] extends Array<Prisma.LogLevel | Prisma.LogDefinition> ? Prisma.GetEvents<ClientOptions['log']> : never : never,
  ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs
> {
  [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['other'] }

    /**
   * ##  Prisma Client ʲˢ
   *
   * Type-safe database client for TypeScript & Node.js
   * @example
   * ```
   * const prisma = new PrismaClient()
   * // Fetch zero or more Users
   * const users = await prisma.user.findMany()
   * ```
   *
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client).
   */

  constructor(optionsArg ?: Prisma.Subset<ClientOptions, Prisma.PrismaClientOptions>);
  $on<V extends (U | 'beforeExit')>(eventType: V, callback: (event: V extends 'query' ? Prisma.QueryEvent : V extends 'beforeExit' ? () => $Utils.JsPromise<void> : Prisma.LogEvent) => void): PrismaClient;

  /**
   * Connect with the database
   */
  $connect(): $Utils.JsPromise<void>;

  /**
   * Disconnect from the database
   */
  $disconnect(): $Utils.JsPromise<void>;

/**
   * Executes a prepared raw query and returns the number of affected rows.
   * @example
   * ```
   * const result = await prisma.$executeRaw`UPDATE User SET cool = ${true} WHERE email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Executes a raw query and returns the number of affected rows.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$executeRawUnsafe('UPDATE User SET cool = $1 WHERE email = $2 ;', true, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $executeRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<number>;

  /**
   * Performs a prepared raw query and returns the `SELECT` data.
   * @example
   * ```
   * const result = await prisma.$queryRaw`SELECT * FROM User WHERE id = ${1} OR email = ${'user@email.com'};`
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRaw<T = unknown>(query: TemplateStringsArray | Prisma.Sql, ...values: any[]): Prisma.PrismaPromise<T>;

  /**
   * Performs a raw query and returns the `SELECT` data.
   * Susceptible to SQL injections, see documentation.
   * @example
   * ```
   * const result = await prisma.$queryRawUnsafe('SELECT * FROM User WHERE id = $1 OR email = $2;', 1, 'user@email.com')
   * ```
   *
   * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/raw-database-access).
   */
  $queryRawUnsafe<T = unknown>(query: string, ...values: any[]): Prisma.PrismaPromise<T>;


  /**
   * Allows the running of a sequence of read/write operations that are guaranteed to either succeed or fail as a whole.
   * @example
   * ```
   * const [george, bob, alice] = await prisma.$transaction([
   *   prisma.user.create({ data: { name: 'George' } }),
   *   prisma.user.create({ data: { name: 'Bob' } }),
   *   prisma.user.create({ data: { name: 'Alice' } }),
   * ])
   * ```
   * 
   * Read more in our [docs](https://www.prisma.io/docs/concepts/components/prisma-client/transactions).
   */
  $transaction<P extends Prisma.PrismaPromise<any>[]>(arg: [...P], options?: { isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<runtime.Types.Utils.UnwrapTuple<P>>

  $transaction<R>(fn: (prisma: Omit<PrismaClient, runtime.ITXClientDenyList>) => $Utils.JsPromise<R>, options?: { maxWait?: number, timeout?: number, isolationLevel?: Prisma.TransactionIsolationLevel }): $Utils.JsPromise<R>


  $extends: $Extensions.ExtendsHook<"extends", Prisma.TypeMapCb<ClientOptions>, ExtArgs, $Utils.Call<Prisma.TypeMapCb<ClientOptions>, {
    extArgs: ExtArgs
  }>>

      /**
   * `prisma.user`: Exposes CRUD operations for the **User** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more Users
    * const users = await prisma.user.findMany()
    * ```
    */
  get user(): Prisma.UserDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.dataBatch`: Exposes CRUD operations for the **DataBatch** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more DataBatches
    * const dataBatches = await prisma.dataBatch.findMany()
    * ```
    */
  get dataBatch(): Prisma.DataBatchDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.dataSample`: Exposes CRUD operations for the **DataSample** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more DataSamples
    * const dataSamples = await prisma.dataSample.findMany()
    * ```
    */
  get dataSample(): Prisma.DataSampleDelegate<ExtArgs, ClientOptions>;

  /**
   * `prisma.imageCapture`: Exposes CRUD operations for the **ImageCapture** model.
    * Example usage:
    * ```ts
    * // Fetch zero or more ImageCaptures
    * const imageCaptures = await prisma.imageCapture.findMany()
    * ```
    */
  get imageCapture(): Prisma.ImageCaptureDelegate<ExtArgs, ClientOptions>;
}

export namespace Prisma {
  export import DMMF = runtime.DMMF

  export type PrismaPromise<T> = $Public.PrismaPromise<T>

  /**
   * Validator
   */
  export import validator = runtime.Public.validator

  /**
   * Prisma Errors
   */
  export import PrismaClientKnownRequestError = runtime.PrismaClientKnownRequestError
  export import PrismaClientUnknownRequestError = runtime.PrismaClientUnknownRequestError
  export import PrismaClientRustPanicError = runtime.PrismaClientRustPanicError
  export import PrismaClientInitializationError = runtime.PrismaClientInitializationError
  export import PrismaClientValidationError = runtime.PrismaClientValidationError

  /**
   * Re-export of sql-template-tag
   */
  export import sql = runtime.sqltag
  export import empty = runtime.empty
  export import join = runtime.join
  export import raw = runtime.raw
  export import Sql = runtime.Sql



  /**
   * Decimal.js
   */
  export import Decimal = runtime.Decimal

  export type DecimalJsLike = runtime.DecimalJsLike

  /**
   * Metrics
   */
  export type Metrics = runtime.Metrics
  export type Metric<T> = runtime.Metric<T>
  export type MetricHistogram = runtime.MetricHistogram
  export type MetricHistogramBucket = runtime.MetricHistogramBucket

  /**
  * Extensions
  */
  export import Extension = $Extensions.UserArgs
  export import getExtensionContext = runtime.Extensions.getExtensionContext
  export import Args = $Public.Args
  export import Payload = $Public.Payload
  export import Result = $Public.Result
  export import Exact = $Public.Exact

  /**
   * Prisma Client JS version: 6.17.1
   * Query Engine version: 272a37d34178c2894197e17273bf937f25acdeac
   */
  export type PrismaVersion = {
    client: string
  }

  export const prismaVersion: PrismaVersion

  /**
   * Utility Types
   */


  export import JsonObject = runtime.JsonObject
  export import JsonArray = runtime.JsonArray
  export import JsonValue = runtime.JsonValue
  export import InputJsonObject = runtime.InputJsonObject
  export import InputJsonArray = runtime.InputJsonArray
  export import InputJsonValue = runtime.InputJsonValue

  /**
   * Types of the values used to represent different kinds of `null` values when working with JSON fields.
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  namespace NullTypes {
    /**
    * Type of `Prisma.DbNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.DbNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class DbNull {
      private DbNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.JsonNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.JsonNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class JsonNull {
      private JsonNull: never
      private constructor()
    }

    /**
    * Type of `Prisma.AnyNull`.
    *
    * You cannot use other instances of this class. Please use the `Prisma.AnyNull` value.
    *
    * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
    */
    class AnyNull {
      private AnyNull: never
      private constructor()
    }
  }

  /**
   * Helper for filtering JSON entries that have `null` on the database (empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const DbNull: NullTypes.DbNull

  /**
   * Helper for filtering JSON entries that have JSON `null` values (not empty on the db)
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const JsonNull: NullTypes.JsonNull

  /**
   * Helper for filtering JSON entries that are `Prisma.DbNull` or `Prisma.JsonNull`
   *
   * @see https://www.prisma.io/docs/concepts/components/prisma-client/working-with-fields/working-with-json-fields#filtering-on-a-json-field
   */
  export const AnyNull: NullTypes.AnyNull

  type SelectAndInclude = {
    select: any
    include: any
  }

  type SelectAndOmit = {
    select: any
    omit: any
  }

  /**
   * Get the type of the value, that the Promise holds.
   */
  export type PromiseType<T extends PromiseLike<any>> = T extends PromiseLike<infer U> ? U : T;

  /**
   * Get the return type of a function which returns a Promise.
   */
  export type PromiseReturnType<T extends (...args: any) => $Utils.JsPromise<any>> = PromiseType<ReturnType<T>>

  /**
   * From T, pick a set of properties whose keys are in the union K
   */
  type Prisma__Pick<T, K extends keyof T> = {
      [P in K]: T[P];
  };


  export type Enumerable<T> = T | Array<T>;

  export type RequiredKeys<T> = {
    [K in keyof T]-?: {} extends Prisma__Pick<T, K> ? never : K
  }[keyof T]

  export type TruthyKeys<T> = keyof {
    [K in keyof T as T[K] extends false | undefined | null ? never : K]: K
  }

  export type TrueKeys<T> = TruthyKeys<Prisma__Pick<T, RequiredKeys<T>>>

  /**
   * Subset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection
   */
  export type Subset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never;
  };

  /**
   * SelectSubset
   * @desc From `T` pick properties that exist in `U`. Simple version of Intersection.
   * Additionally, it validates, if both select and include are present. If the case, it errors.
   */
  export type SelectSubset<T, U> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    (T extends SelectAndInclude
      ? 'Please either choose `select` or `include`.'
      : T extends SelectAndOmit
        ? 'Please either choose `select` or `omit`.'
        : {})

  /**
   * Subset + Intersection
   * @desc From `T` pick properties that exist in `U` and intersect `K`
   */
  export type SubsetIntersection<T, U, K> = {
    [key in keyof T]: key extends keyof U ? T[key] : never
  } &
    K

  type Without<T, U> = { [P in Exclude<keyof T, keyof U>]?: never };

  /**
   * XOR is needed to have a real mutually exclusive union type
   * https://stackoverflow.com/questions/42123407/does-typescript-support-mutually-exclusive-types
   */
  type XOR<T, U> =
    T extends object ?
    U extends object ?
      (Without<T, U> & U) | (Without<U, T> & T)
    : U : T


  /**
   * Is T a Record?
   */
  type IsObject<T extends any> = T extends Array<any>
  ? False
  : T extends Date
  ? False
  : T extends Uint8Array
  ? False
  : T extends BigInt
  ? False
  : T extends object
  ? True
  : False


  /**
   * If it's T[], return T
   */
  export type UnEnumerate<T extends unknown> = T extends Array<infer U> ? U : T

  /**
   * From ts-toolbelt
   */

  type __Either<O extends object, K extends Key> = Omit<O, K> &
    {
      // Merge all but K
      [P in K]: Prisma__Pick<O, P & keyof O> // With K possibilities
    }[K]

  type EitherStrict<O extends object, K extends Key> = Strict<__Either<O, K>>

  type EitherLoose<O extends object, K extends Key> = ComputeRaw<__Either<O, K>>

  type _Either<
    O extends object,
    K extends Key,
    strict extends Boolean
  > = {
    1: EitherStrict<O, K>
    0: EitherLoose<O, K>
  }[strict]

  type Either<
    O extends object,
    K extends Key,
    strict extends Boolean = 1
  > = O extends unknown ? _Either<O, K, strict> : never

  export type Union = any

  type PatchUndefined<O extends object, O1 extends object> = {
    [K in keyof O]: O[K] extends undefined ? At<O1, K> : O[K]
  } & {}

  /** Helper Types for "Merge" **/
  export type IntersectOf<U extends Union> = (
    U extends unknown ? (k: U) => void : never
  ) extends (k: infer I) => void
    ? I
    : never

  export type Overwrite<O extends object, O1 extends object> = {
      [K in keyof O]: K extends keyof O1 ? O1[K] : O[K];
  } & {};

  type _Merge<U extends object> = IntersectOf<Overwrite<U, {
      [K in keyof U]-?: At<U, K>;
  }>>;

  type Key = string | number | symbol;
  type AtBasic<O extends object, K extends Key> = K extends keyof O ? O[K] : never;
  type AtStrict<O extends object, K extends Key> = O[K & keyof O];
  type AtLoose<O extends object, K extends Key> = O extends unknown ? AtStrict<O, K> : never;
  export type At<O extends object, K extends Key, strict extends Boolean = 1> = {
      1: AtStrict<O, K>;
      0: AtLoose<O, K>;
  }[strict];

  export type ComputeRaw<A extends any> = A extends Function ? A : {
    [K in keyof A]: A[K];
  } & {};

  export type OptionalFlat<O> = {
    [K in keyof O]?: O[K];
  } & {};

  type _Record<K extends keyof any, T> = {
    [P in K]: T;
  };

  // cause typescript not to expand types and preserve names
  type NoExpand<T> = T extends unknown ? T : never;

  // this type assumes the passed object is entirely optional
  type AtLeast<O extends object, K extends string> = NoExpand<
    O extends unknown
    ? | (K extends keyof O ? { [P in K]: O[P] } & O : O)
      | {[P in keyof O as P extends K ? P : never]-?: O[P]} & O
    : never>;

  type _Strict<U, _U = U> = U extends unknown ? U & OptionalFlat<_Record<Exclude<Keys<_U>, keyof U>, never>> : never;

  export type Strict<U extends object> = ComputeRaw<_Strict<U>>;
  /** End Helper Types for "Merge" **/

  export type Merge<U extends object> = ComputeRaw<_Merge<Strict<U>>>;

  /**
  A [[Boolean]]
  */
  export type Boolean = True | False

  // /**
  // 1
  // */
  export type True = 1

  /**
  0
  */
  export type False = 0

  export type Not<B extends Boolean> = {
    0: 1
    1: 0
  }[B]

  export type Extends<A1 extends any, A2 extends any> = [A1] extends [never]
    ? 0 // anything `never` is false
    : A1 extends A2
    ? 1
    : 0

  export type Has<U extends Union, U1 extends Union> = Not<
    Extends<Exclude<U1, U>, U1>
  >

  export type Or<B1 extends Boolean, B2 extends Boolean> = {
    0: {
      0: 0
      1: 1
    }
    1: {
      0: 1
      1: 1
    }
  }[B1][B2]

  export type Keys<U extends Union> = U extends unknown ? keyof U : never

  type Cast<A, B> = A extends B ? A : B;

  export const type: unique symbol;



  /**
   * Used by group by
   */

  export type GetScalarType<T, O> = O extends object ? {
    [P in keyof T]: P extends keyof O
      ? O[P]
      : never
  } : never

  type FieldPaths<
    T,
    U = Omit<T, '_avg' | '_sum' | '_count' | '_min' | '_max'>
  > = IsObject<T> extends True ? U : T

  type GetHavingFields<T> = {
    [K in keyof T]: Or<
      Or<Extends<'OR', K>, Extends<'AND', K>>,
      Extends<'NOT', K>
    > extends True
      ? // infer is only needed to not hit TS limit
        // based on the brilliant idea of Pierre-Antoine Mills
        // https://github.com/microsoft/TypeScript/issues/30188#issuecomment-478938437
        T[K] extends infer TK
        ? GetHavingFields<UnEnumerate<TK> extends object ? Merge<UnEnumerate<TK>> : never>
        : never
      : {} extends FieldPaths<T[K]>
      ? never
      : K
  }[keyof T]

  /**
   * Convert tuple to union
   */
  type _TupleToUnion<T> = T extends (infer E)[] ? E : never
  type TupleToUnion<K extends readonly any[]> = _TupleToUnion<K>
  type MaybeTupleToUnion<T> = T extends any[] ? TupleToUnion<T> : T

  /**
   * Like `Pick`, but additionally can also accept an array of keys
   */
  type PickEnumerable<T, K extends Enumerable<keyof T> | keyof T> = Prisma__Pick<T, MaybeTupleToUnion<K>>

  /**
   * Exclude all keys with underscores
   */
  type ExcludeUnderscoreKeys<T extends string> = T extends `_${string}` ? never : T


  export type FieldRef<Model, FieldType> = runtime.FieldRef<Model, FieldType>

  type FieldRefInputType<Model, FieldType> = Model extends never ? never : FieldRef<Model, FieldType>


  export const ModelName: {
    User: 'User',
    DataBatch: 'DataBatch',
    DataSample: 'DataSample',
    ImageCapture: 'ImageCapture'
  };

  export type ModelName = (typeof ModelName)[keyof typeof ModelName]


  export type Datasources = {
    db?: Datasource
  }

  interface TypeMapCb<ClientOptions = {}> extends $Utils.Fn<{extArgs: $Extensions.InternalArgs }, $Utils.Record<string, any>> {
    returns: Prisma.TypeMap<this['params']['extArgs'], ClientOptions extends { omit: infer OmitOptions } ? OmitOptions : {}>
  }

  export type TypeMap<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> = {
    globalOmitOptions: {
      omit: GlobalOmitOptions
    }
    meta: {
      modelProps: "user" | "dataBatch" | "dataSample" | "imageCapture"
      txIsolationLevel: Prisma.TransactionIsolationLevel
    }
    model: {
      User: {
        payload: Prisma.$UserPayload<ExtArgs>
        fields: Prisma.UserFieldRefs
        operations: {
          findUnique: {
            args: Prisma.UserFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.UserFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          findFirst: {
            args: Prisma.UserFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.UserFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          findMany: {
            args: Prisma.UserFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          create: {
            args: Prisma.UserCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          createMany: {
            args: Prisma.UserCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.UserCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          delete: {
            args: Prisma.UserDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          update: {
            args: Prisma.UserUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          deleteMany: {
            args: Prisma.UserDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.UserUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.UserUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>[]
          }
          upsert: {
            args: Prisma.UserUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$UserPayload>
          }
          aggregate: {
            args: Prisma.UserAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateUser>
          }
          groupBy: {
            args: Prisma.UserGroupByArgs<ExtArgs>
            result: $Utils.Optional<UserGroupByOutputType>[]
          }
          count: {
            args: Prisma.UserCountArgs<ExtArgs>
            result: $Utils.Optional<UserCountAggregateOutputType> | number
          }
        }
      }
      DataBatch: {
        payload: Prisma.$DataBatchPayload<ExtArgs>
        fields: Prisma.DataBatchFieldRefs
        operations: {
          findUnique: {
            args: Prisma.DataBatchFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.DataBatchFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>
          }
          findFirst: {
            args: Prisma.DataBatchFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.DataBatchFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>
          }
          findMany: {
            args: Prisma.DataBatchFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>[]
          }
          create: {
            args: Prisma.DataBatchCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>
          }
          createMany: {
            args: Prisma.DataBatchCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.DataBatchCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>[]
          }
          delete: {
            args: Prisma.DataBatchDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>
          }
          update: {
            args: Prisma.DataBatchUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>
          }
          deleteMany: {
            args: Prisma.DataBatchDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.DataBatchUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.DataBatchUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>[]
          }
          upsert: {
            args: Prisma.DataBatchUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataBatchPayload>
          }
          aggregate: {
            args: Prisma.DataBatchAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateDataBatch>
          }
          groupBy: {
            args: Prisma.DataBatchGroupByArgs<ExtArgs>
            result: $Utils.Optional<DataBatchGroupByOutputType>[]
          }
          count: {
            args: Prisma.DataBatchCountArgs<ExtArgs>
            result: $Utils.Optional<DataBatchCountAggregateOutputType> | number
          }
        }
      }
      DataSample: {
        payload: Prisma.$DataSamplePayload<ExtArgs>
        fields: Prisma.DataSampleFieldRefs
        operations: {
          findUnique: {
            args: Prisma.DataSampleFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.DataSampleFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>
          }
          findFirst: {
            args: Prisma.DataSampleFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.DataSampleFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>
          }
          findMany: {
            args: Prisma.DataSampleFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>[]
          }
          create: {
            args: Prisma.DataSampleCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>
          }
          createMany: {
            args: Prisma.DataSampleCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.DataSampleCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>[]
          }
          delete: {
            args: Prisma.DataSampleDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>
          }
          update: {
            args: Prisma.DataSampleUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>
          }
          deleteMany: {
            args: Prisma.DataSampleDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.DataSampleUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.DataSampleUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>[]
          }
          upsert: {
            args: Prisma.DataSampleUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$DataSamplePayload>
          }
          aggregate: {
            args: Prisma.DataSampleAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateDataSample>
          }
          groupBy: {
            args: Prisma.DataSampleGroupByArgs<ExtArgs>
            result: $Utils.Optional<DataSampleGroupByOutputType>[]
          }
          count: {
            args: Prisma.DataSampleCountArgs<ExtArgs>
            result: $Utils.Optional<DataSampleCountAggregateOutputType> | number
          }
        }
      }
      ImageCapture: {
        payload: Prisma.$ImageCapturePayload<ExtArgs>
        fields: Prisma.ImageCaptureFieldRefs
        operations: {
          findUnique: {
            args: Prisma.ImageCaptureFindUniqueArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload> | null
          }
          findUniqueOrThrow: {
            args: Prisma.ImageCaptureFindUniqueOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>
          }
          findFirst: {
            args: Prisma.ImageCaptureFindFirstArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload> | null
          }
          findFirstOrThrow: {
            args: Prisma.ImageCaptureFindFirstOrThrowArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>
          }
          findMany: {
            args: Prisma.ImageCaptureFindManyArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>[]
          }
          create: {
            args: Prisma.ImageCaptureCreateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>
          }
          createMany: {
            args: Prisma.ImageCaptureCreateManyArgs<ExtArgs>
            result: BatchPayload
          }
          createManyAndReturn: {
            args: Prisma.ImageCaptureCreateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>[]
          }
          delete: {
            args: Prisma.ImageCaptureDeleteArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>
          }
          update: {
            args: Prisma.ImageCaptureUpdateArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>
          }
          deleteMany: {
            args: Prisma.ImageCaptureDeleteManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateMany: {
            args: Prisma.ImageCaptureUpdateManyArgs<ExtArgs>
            result: BatchPayload
          }
          updateManyAndReturn: {
            args: Prisma.ImageCaptureUpdateManyAndReturnArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>[]
          }
          upsert: {
            args: Prisma.ImageCaptureUpsertArgs<ExtArgs>
            result: $Utils.PayloadToResult<Prisma.$ImageCapturePayload>
          }
          aggregate: {
            args: Prisma.ImageCaptureAggregateArgs<ExtArgs>
            result: $Utils.Optional<AggregateImageCapture>
          }
          groupBy: {
            args: Prisma.ImageCaptureGroupByArgs<ExtArgs>
            result: $Utils.Optional<ImageCaptureGroupByOutputType>[]
          }
          count: {
            args: Prisma.ImageCaptureCountArgs<ExtArgs>
            result: $Utils.Optional<ImageCaptureCountAggregateOutputType> | number
          }
        }
      }
    }
  } & {
    other: {
      payload: any
      operations: {
        $executeRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $executeRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
        $queryRaw: {
          args: [query: TemplateStringsArray | Prisma.Sql, ...values: any[]],
          result: any
        }
        $queryRawUnsafe: {
          args: [query: string, ...values: any[]],
          result: any
        }
      }
    }
  }
  export const defineExtension: $Extensions.ExtendsHook<"define", Prisma.TypeMapCb, $Extensions.DefaultArgs>
  export type DefaultPrismaClient = PrismaClient
  export type ErrorFormat = 'pretty' | 'colorless' | 'minimal'
  export interface PrismaClientOptions {
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasources?: Datasources
    /**
     * Overwrites the datasource url from your schema.prisma file
     */
    datasourceUrl?: string
    /**
     * @default "colorless"
     */
    errorFormat?: ErrorFormat
    /**
     * @example
     * ```
     * // Shorthand for `emit: 'stdout'`
     * log: ['query', 'info', 'warn', 'error']
     * 
     * // Emit as events only
     * log: [
     *   { emit: 'event', level: 'query' },
     *   { emit: 'event', level: 'info' },
     *   { emit: 'event', level: 'warn' }
     *   { emit: 'event', level: 'error' }
     * ]
     * 
     * / Emit as events and log to stdout
     * og: [
     *  { emit: 'stdout', level: 'query' },
     *  { emit: 'stdout', level: 'info' },
     *  { emit: 'stdout', level: 'warn' }
     *  { emit: 'stdout', level: 'error' }
     * 
     * ```
     * Read more in our [docs](https://www.prisma.io/docs/reference/tools-and-interfaces/prisma-client/logging#the-log-option).
     */
    log?: (LogLevel | LogDefinition)[]
    /**
     * The default values for transactionOptions
     * maxWait ?= 2000
     * timeout ?= 5000
     */
    transactionOptions?: {
      maxWait?: number
      timeout?: number
      isolationLevel?: Prisma.TransactionIsolationLevel
    }
    /**
     * Global configuration for omitting model fields by default.
     * 
     * @example
     * ```
     * const prisma = new PrismaClient({
     *   omit: {
     *     user: {
     *       password: true
     *     }
     *   }
     * })
     * ```
     */
    omit?: Prisma.GlobalOmitConfig
  }
  export type GlobalOmitConfig = {
    user?: UserOmit
    dataBatch?: DataBatchOmit
    dataSample?: DataSampleOmit
    imageCapture?: ImageCaptureOmit
  }

  /* Types for Logging */
  export type LogLevel = 'info' | 'query' | 'warn' | 'error'
  export type LogDefinition = {
    level: LogLevel
    emit: 'stdout' | 'event'
  }

  export type CheckIsLogLevel<T> = T extends LogLevel ? T : never;

  export type GetLogType<T> = CheckIsLogLevel<
    T extends LogDefinition ? T['level'] : T
  >;

  export type GetEvents<T extends any[]> = T extends Array<LogLevel | LogDefinition>
    ? GetLogType<T[number]>
    : never;

  export type QueryEvent = {
    timestamp: Date
    query: string
    params: string
    duration: number
    target: string
  }

  export type LogEvent = {
    timestamp: Date
    message: string
    target: string
  }
  /* End Types for Logging */


  export type PrismaAction =
    | 'findUnique'
    | 'findUniqueOrThrow'
    | 'findMany'
    | 'findFirst'
    | 'findFirstOrThrow'
    | 'create'
    | 'createMany'
    | 'createManyAndReturn'
    | 'update'
    | 'updateMany'
    | 'updateManyAndReturn'
    | 'upsert'
    | 'delete'
    | 'deleteMany'
    | 'executeRaw'
    | 'queryRaw'
    | 'aggregate'
    | 'count'
    | 'runCommandRaw'
    | 'findRaw'
    | 'groupBy'

  // tested in getLogLevel.test.ts
  export function getLogLevel(log: Array<LogLevel | LogDefinition>): LogLevel | undefined;

  /**
   * `PrismaClient` proxy available in interactive transactions.
   */
  export type TransactionClient = Omit<Prisma.DefaultPrismaClient, runtime.ITXClientDenyList>

  export type Datasource = {
    url?: string
  }

  /**
   * Count Types
   */


  /**
   * Count Type DataBatchCountOutputType
   */

  export type DataBatchCountOutputType = {
    samples: number
  }

  export type DataBatchCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    samples?: boolean | DataBatchCountOutputTypeCountSamplesArgs
  }

  // Custom InputTypes
  /**
   * DataBatchCountOutputType without action
   */
  export type DataBatchCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatchCountOutputType
     */
    select?: DataBatchCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * DataBatchCountOutputType without action
   */
  export type DataBatchCountOutputTypeCountSamplesArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: DataSampleWhereInput
  }


  /**
   * Count Type DataSampleCountOutputType
   */

  export type DataSampleCountOutputType = {
    images: number
  }

  export type DataSampleCountOutputTypeSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    images?: boolean | DataSampleCountOutputTypeCountImagesArgs
  }

  // Custom InputTypes
  /**
   * DataSampleCountOutputType without action
   */
  export type DataSampleCountOutputTypeDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSampleCountOutputType
     */
    select?: DataSampleCountOutputTypeSelect<ExtArgs> | null
  }

  /**
   * DataSampleCountOutputType without action
   */
  export type DataSampleCountOutputTypeCountImagesArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: ImageCaptureWhereInput
  }


  /**
   * Models
   */

  /**
   * Model User
   */

  export type AggregateUser = {
    _count: UserCountAggregateOutputType | null
    _avg: UserAvgAggregateOutputType | null
    _sum: UserSumAggregateOutputType | null
    _min: UserMinAggregateOutputType | null
    _max: UserMaxAggregateOutputType | null
  }

  export type UserAvgAggregateOutputType = {
    id: number | null
  }

  export type UserSumAggregateOutputType = {
    id: number | null
  }

  export type UserMinAggregateOutputType = {
    id: number | null
    email: string | null
    password: string | null
    name: string | null
    createdAt: Date | null
  }

  export type UserMaxAggregateOutputType = {
    id: number | null
    email: string | null
    password: string | null
    name: string | null
    createdAt: Date | null
  }

  export type UserCountAggregateOutputType = {
    id: number
    email: number
    password: number
    name: number
    createdAt: number
    _all: number
  }


  export type UserAvgAggregateInputType = {
    id?: true
  }

  export type UserSumAggregateInputType = {
    id?: true
  }

  export type UserMinAggregateInputType = {
    id?: true
    email?: true
    password?: true
    name?: true
    createdAt?: true
  }

  export type UserMaxAggregateInputType = {
    id?: true
    email?: true
    password?: true
    name?: true
    createdAt?: true
  }

  export type UserCountAggregateInputType = {
    id?: true
    email?: true
    password?: true
    name?: true
    createdAt?: true
    _all?: true
  }

  export type UserAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which User to aggregate.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned Users
    **/
    _count?: true | UserCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: UserAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: UserSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: UserMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: UserMaxAggregateInputType
  }

  export type GetUserAggregateType<T extends UserAggregateArgs> = {
        [P in keyof T & keyof AggregateUser]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateUser[P]>
      : GetScalarType<T[P], AggregateUser[P]>
  }




  export type UserGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: UserWhereInput
    orderBy?: UserOrderByWithAggregationInput | UserOrderByWithAggregationInput[]
    by: UserScalarFieldEnum[] | UserScalarFieldEnum
    having?: UserScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: UserCountAggregateInputType | true
    _avg?: UserAvgAggregateInputType
    _sum?: UserSumAggregateInputType
    _min?: UserMinAggregateInputType
    _max?: UserMaxAggregateInputType
  }

  export type UserGroupByOutputType = {
    id: number
    email: string
    password: string
    name: string | null
    createdAt: Date
    _count: UserCountAggregateOutputType | null
    _avg: UserAvgAggregateOutputType | null
    _sum: UserSumAggregateOutputType | null
    _min: UserMinAggregateOutputType | null
    _max: UserMaxAggregateOutputType | null
  }

  type GetUserGroupByPayload<T extends UserGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<UserGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof UserGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], UserGroupByOutputType[P]>
            : GetScalarType<T[P], UserGroupByOutputType[P]>
        }
      >
    >


  export type UserSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    password?: boolean
    name?: boolean
    createdAt?: boolean
  }, ExtArgs["result"]["user"]>

  export type UserSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    password?: boolean
    name?: boolean
    createdAt?: boolean
  }, ExtArgs["result"]["user"]>

  export type UserSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    email?: boolean
    password?: boolean
    name?: boolean
    createdAt?: boolean
  }, ExtArgs["result"]["user"]>

  export type UserSelectScalar = {
    id?: boolean
    email?: boolean
    password?: boolean
    name?: boolean
    createdAt?: boolean
  }

  export type UserOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "email" | "password" | "name" | "createdAt", ExtArgs["result"]["user"]>

  export type $UserPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "User"
    objects: {}
    scalars: $Extensions.GetPayloadResult<{
      id: number
      email: string
      password: string
      name: string | null
      createdAt: Date
    }, ExtArgs["result"]["user"]>
    composites: {}
  }

  type UserGetPayload<S extends boolean | null | undefined | UserDefaultArgs> = $Result.GetResult<Prisma.$UserPayload, S>

  type UserCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<UserFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: UserCountAggregateInputType | true
    }

  export interface UserDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['User'], meta: { name: 'User' } }
    /**
     * Find zero or one User that matches the filter.
     * @param {UserFindUniqueArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends UserFindUniqueArgs>(args: SelectSubset<T, UserFindUniqueArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one User that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {UserFindUniqueOrThrowArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends UserFindUniqueOrThrowArgs>(args: SelectSubset<T, UserFindUniqueOrThrowArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first User that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindFirstArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends UserFindFirstArgs>(args?: SelectSubset<T, UserFindFirstArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first User that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindFirstOrThrowArgs} args - Arguments to find a User
     * @example
     * // Get one User
     * const user = await prisma.user.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends UserFindFirstOrThrowArgs>(args?: SelectSubset<T, UserFindFirstOrThrowArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more Users that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all Users
     * const users = await prisma.user.findMany()
     * 
     * // Get first 10 Users
     * const users = await prisma.user.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const userWithIdOnly = await prisma.user.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends UserFindManyArgs>(args?: SelectSubset<T, UserFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a User.
     * @param {UserCreateArgs} args - Arguments to create a User.
     * @example
     * // Create one User
     * const User = await prisma.user.create({
     *   data: {
     *     // ... data to create a User
     *   }
     * })
     * 
     */
    create<T extends UserCreateArgs>(args: SelectSubset<T, UserCreateArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many Users.
     * @param {UserCreateManyArgs} args - Arguments to create many Users.
     * @example
     * // Create many Users
     * const user = await prisma.user.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends UserCreateManyArgs>(args?: SelectSubset<T, UserCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many Users and returns the data saved in the database.
     * @param {UserCreateManyAndReturnArgs} args - Arguments to create many Users.
     * @example
     * // Create many Users
     * const user = await prisma.user.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many Users and only return the `id`
     * const userWithIdOnly = await prisma.user.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends UserCreateManyAndReturnArgs>(args?: SelectSubset<T, UserCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a User.
     * @param {UserDeleteArgs} args - Arguments to delete one User.
     * @example
     * // Delete one User
     * const User = await prisma.user.delete({
     *   where: {
     *     // ... filter to delete one User
     *   }
     * })
     * 
     */
    delete<T extends UserDeleteArgs>(args: SelectSubset<T, UserDeleteArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one User.
     * @param {UserUpdateArgs} args - Arguments to update one User.
     * @example
     * // Update one User
     * const user = await prisma.user.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends UserUpdateArgs>(args: SelectSubset<T, UserUpdateArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more Users.
     * @param {UserDeleteManyArgs} args - Arguments to filter Users to delete.
     * @example
     * // Delete a few Users
     * const { count } = await prisma.user.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends UserDeleteManyArgs>(args?: SelectSubset<T, UserDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Users.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many Users
     * const user = await prisma.user.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends UserUpdateManyArgs>(args: SelectSubset<T, UserUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more Users and returns the data updated in the database.
     * @param {UserUpdateManyAndReturnArgs} args - Arguments to update many Users.
     * @example
     * // Update many Users
     * const user = await prisma.user.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more Users and only return the `id`
     * const userWithIdOnly = await prisma.user.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends UserUpdateManyAndReturnArgs>(args: SelectSubset<T, UserUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one User.
     * @param {UserUpsertArgs} args - Arguments to update or create a User.
     * @example
     * // Update or create a User
     * const user = await prisma.user.upsert({
     *   create: {
     *     // ... data to create a User
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the User we want to update
     *   }
     * })
     */
    upsert<T extends UserUpsertArgs>(args: SelectSubset<T, UserUpsertArgs<ExtArgs>>): Prisma__UserClient<$Result.GetResult<Prisma.$UserPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of Users.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserCountArgs} args - Arguments to filter Users to count.
     * @example
     * // Count the number of Users
     * const count = await prisma.user.count({
     *   where: {
     *     // ... the filter for the Users we want to count
     *   }
     * })
    **/
    count<T extends UserCountArgs>(
      args?: Subset<T, UserCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], UserCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a User.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends UserAggregateArgs>(args: Subset<T, UserAggregateArgs>): Prisma.PrismaPromise<GetUserAggregateType<T>>

    /**
     * Group by User.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {UserGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends UserGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: UserGroupByArgs['orderBy'] }
        : { orderBy?: UserGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, UserGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetUserGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the User model
   */
  readonly fields: UserFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for User.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__UserClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the User model
   */
  interface UserFieldRefs {
    readonly id: FieldRef<"User", 'Int'>
    readonly email: FieldRef<"User", 'String'>
    readonly password: FieldRef<"User", 'String'>
    readonly name: FieldRef<"User", 'String'>
    readonly createdAt: FieldRef<"User", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * User findUnique
   */
  export type UserFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User findUniqueOrThrow
   */
  export type UserFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User findFirst
   */
  export type UserFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Users.
     */
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User findFirstOrThrow
   */
  export type UserFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Filter, which User to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of Users.
     */
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User findMany
   */
  export type UserFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Filter, which Users to fetch.
     */
    where?: UserWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of Users to fetch.
     */
    orderBy?: UserOrderByWithRelationInput | UserOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing Users.
     */
    cursor?: UserWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` Users from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` Users.
     */
    skip?: number
    distinct?: UserScalarFieldEnum | UserScalarFieldEnum[]
  }

  /**
   * User create
   */
  export type UserCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data needed to create a User.
     */
    data: XOR<UserCreateInput, UserUncheckedCreateInput>
  }

  /**
   * User createMany
   */
  export type UserCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many Users.
     */
    data: UserCreateManyInput | UserCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * User createManyAndReturn
   */
  export type UserCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data used to create many Users.
     */
    data: UserCreateManyInput | UserCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * User update
   */
  export type UserUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data needed to update a User.
     */
    data: XOR<UserUpdateInput, UserUncheckedUpdateInput>
    /**
     * Choose, which User to update.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User updateMany
   */
  export type UserUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update Users.
     */
    data: XOR<UserUpdateManyMutationInput, UserUncheckedUpdateManyInput>
    /**
     * Filter which Users to update
     */
    where?: UserWhereInput
    /**
     * Limit how many Users to update.
     */
    limit?: number
  }

  /**
   * User updateManyAndReturn
   */
  export type UserUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The data used to update Users.
     */
    data: XOR<UserUpdateManyMutationInput, UserUncheckedUpdateManyInput>
    /**
     * Filter which Users to update
     */
    where?: UserWhereInput
    /**
     * Limit how many Users to update.
     */
    limit?: number
  }

  /**
   * User upsert
   */
  export type UserUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * The filter to search for the User to update in case it exists.
     */
    where: UserWhereUniqueInput
    /**
     * In case the User found by the `where` argument doesn't exist, create a new User with this data.
     */
    create: XOR<UserCreateInput, UserUncheckedCreateInput>
    /**
     * In case the User was found with the provided `where` argument, update it with this data.
     */
    update: XOR<UserUpdateInput, UserUncheckedUpdateInput>
  }

  /**
   * User delete
   */
  export type UserDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
    /**
     * Filter which User to delete.
     */
    where: UserWhereUniqueInput
  }

  /**
   * User deleteMany
   */
  export type UserDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which Users to delete
     */
    where?: UserWhereInput
    /**
     * Limit how many Users to delete.
     */
    limit?: number
  }

  /**
   * User without action
   */
  export type UserDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the User
     */
    select?: UserSelect<ExtArgs> | null
    /**
     * Omit specific fields from the User
     */
    omit?: UserOmit<ExtArgs> | null
  }


  /**
   * Model DataBatch
   */

  export type AggregateDataBatch = {
    _count: DataBatchCountAggregateOutputType | null
    _avg: DataBatchAvgAggregateOutputType | null
    _sum: DataBatchSumAggregateOutputType | null
    _min: DataBatchMinAggregateOutputType | null
    _max: DataBatchMaxAggregateOutputType | null
  }

  export type DataBatchAvgAggregateOutputType = {
    id: number | null
  }

  export type DataBatchSumAggregateOutputType = {
    id: number | null
  }

  export type DataBatchMinAggregateOutputType = {
    id: number | null
    name: string | null
    createdAt: Date | null
  }

  export type DataBatchMaxAggregateOutputType = {
    id: number | null
    name: string | null
    createdAt: Date | null
  }

  export type DataBatchCountAggregateOutputType = {
    id: number
    name: number
    createdAt: number
    _all: number
  }


  export type DataBatchAvgAggregateInputType = {
    id?: true
  }

  export type DataBatchSumAggregateInputType = {
    id?: true
  }

  export type DataBatchMinAggregateInputType = {
    id?: true
    name?: true
    createdAt?: true
  }

  export type DataBatchMaxAggregateInputType = {
    id?: true
    name?: true
    createdAt?: true
  }

  export type DataBatchCountAggregateInputType = {
    id?: true
    name?: true
    createdAt?: true
    _all?: true
  }

  export type DataBatchAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which DataBatch to aggregate.
     */
    where?: DataBatchWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataBatches to fetch.
     */
    orderBy?: DataBatchOrderByWithRelationInput | DataBatchOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: DataBatchWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataBatches from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataBatches.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned DataBatches
    **/
    _count?: true | DataBatchCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: DataBatchAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: DataBatchSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: DataBatchMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: DataBatchMaxAggregateInputType
  }

  export type GetDataBatchAggregateType<T extends DataBatchAggregateArgs> = {
        [P in keyof T & keyof AggregateDataBatch]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateDataBatch[P]>
      : GetScalarType<T[P], AggregateDataBatch[P]>
  }




  export type DataBatchGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: DataBatchWhereInput
    orderBy?: DataBatchOrderByWithAggregationInput | DataBatchOrderByWithAggregationInput[]
    by: DataBatchScalarFieldEnum[] | DataBatchScalarFieldEnum
    having?: DataBatchScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: DataBatchCountAggregateInputType | true
    _avg?: DataBatchAvgAggregateInputType
    _sum?: DataBatchSumAggregateInputType
    _min?: DataBatchMinAggregateInputType
    _max?: DataBatchMaxAggregateInputType
  }

  export type DataBatchGroupByOutputType = {
    id: number
    name: string
    createdAt: Date
    _count: DataBatchCountAggregateOutputType | null
    _avg: DataBatchAvgAggregateOutputType | null
    _sum: DataBatchSumAggregateOutputType | null
    _min: DataBatchMinAggregateOutputType | null
    _max: DataBatchMaxAggregateOutputType | null
  }

  type GetDataBatchGroupByPayload<T extends DataBatchGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<DataBatchGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof DataBatchGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], DataBatchGroupByOutputType[P]>
            : GetScalarType<T[P], DataBatchGroupByOutputType[P]>
        }
      >
    >


  export type DataBatchSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    createdAt?: boolean
    samples?: boolean | DataBatch$samplesArgs<ExtArgs>
    _count?: boolean | DataBatchCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["dataBatch"]>

  export type DataBatchSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    createdAt?: boolean
  }, ExtArgs["result"]["dataBatch"]>

  export type DataBatchSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    name?: boolean
    createdAt?: boolean
  }, ExtArgs["result"]["dataBatch"]>

  export type DataBatchSelectScalar = {
    id?: boolean
    name?: boolean
    createdAt?: boolean
  }

  export type DataBatchOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "name" | "createdAt", ExtArgs["result"]["dataBatch"]>
  export type DataBatchInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    samples?: boolean | DataBatch$samplesArgs<ExtArgs>
    _count?: boolean | DataBatchCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type DataBatchIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}
  export type DataBatchIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {}

  export type $DataBatchPayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "DataBatch"
    objects: {
      samples: Prisma.$DataSamplePayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: number
      name: string
      createdAt: Date
    }, ExtArgs["result"]["dataBatch"]>
    composites: {}
  }

  type DataBatchGetPayload<S extends boolean | null | undefined | DataBatchDefaultArgs> = $Result.GetResult<Prisma.$DataBatchPayload, S>

  type DataBatchCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<DataBatchFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: DataBatchCountAggregateInputType | true
    }

  export interface DataBatchDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['DataBatch'], meta: { name: 'DataBatch' } }
    /**
     * Find zero or one DataBatch that matches the filter.
     * @param {DataBatchFindUniqueArgs} args - Arguments to find a DataBatch
     * @example
     * // Get one DataBatch
     * const dataBatch = await prisma.dataBatch.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends DataBatchFindUniqueArgs>(args: SelectSubset<T, DataBatchFindUniqueArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one DataBatch that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {DataBatchFindUniqueOrThrowArgs} args - Arguments to find a DataBatch
     * @example
     * // Get one DataBatch
     * const dataBatch = await prisma.dataBatch.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends DataBatchFindUniqueOrThrowArgs>(args: SelectSubset<T, DataBatchFindUniqueOrThrowArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first DataBatch that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchFindFirstArgs} args - Arguments to find a DataBatch
     * @example
     * // Get one DataBatch
     * const dataBatch = await prisma.dataBatch.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends DataBatchFindFirstArgs>(args?: SelectSubset<T, DataBatchFindFirstArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first DataBatch that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchFindFirstOrThrowArgs} args - Arguments to find a DataBatch
     * @example
     * // Get one DataBatch
     * const dataBatch = await prisma.dataBatch.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends DataBatchFindFirstOrThrowArgs>(args?: SelectSubset<T, DataBatchFindFirstOrThrowArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more DataBatches that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all DataBatches
     * const dataBatches = await prisma.dataBatch.findMany()
     * 
     * // Get first 10 DataBatches
     * const dataBatches = await prisma.dataBatch.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const dataBatchWithIdOnly = await prisma.dataBatch.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends DataBatchFindManyArgs>(args?: SelectSubset<T, DataBatchFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a DataBatch.
     * @param {DataBatchCreateArgs} args - Arguments to create a DataBatch.
     * @example
     * // Create one DataBatch
     * const DataBatch = await prisma.dataBatch.create({
     *   data: {
     *     // ... data to create a DataBatch
     *   }
     * })
     * 
     */
    create<T extends DataBatchCreateArgs>(args: SelectSubset<T, DataBatchCreateArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many DataBatches.
     * @param {DataBatchCreateManyArgs} args - Arguments to create many DataBatches.
     * @example
     * // Create many DataBatches
     * const dataBatch = await prisma.dataBatch.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends DataBatchCreateManyArgs>(args?: SelectSubset<T, DataBatchCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many DataBatches and returns the data saved in the database.
     * @param {DataBatchCreateManyAndReturnArgs} args - Arguments to create many DataBatches.
     * @example
     * // Create many DataBatches
     * const dataBatch = await prisma.dataBatch.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many DataBatches and only return the `id`
     * const dataBatchWithIdOnly = await prisma.dataBatch.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends DataBatchCreateManyAndReturnArgs>(args?: SelectSubset<T, DataBatchCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a DataBatch.
     * @param {DataBatchDeleteArgs} args - Arguments to delete one DataBatch.
     * @example
     * // Delete one DataBatch
     * const DataBatch = await prisma.dataBatch.delete({
     *   where: {
     *     // ... filter to delete one DataBatch
     *   }
     * })
     * 
     */
    delete<T extends DataBatchDeleteArgs>(args: SelectSubset<T, DataBatchDeleteArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one DataBatch.
     * @param {DataBatchUpdateArgs} args - Arguments to update one DataBatch.
     * @example
     * // Update one DataBatch
     * const dataBatch = await prisma.dataBatch.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends DataBatchUpdateArgs>(args: SelectSubset<T, DataBatchUpdateArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more DataBatches.
     * @param {DataBatchDeleteManyArgs} args - Arguments to filter DataBatches to delete.
     * @example
     * // Delete a few DataBatches
     * const { count } = await prisma.dataBatch.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends DataBatchDeleteManyArgs>(args?: SelectSubset<T, DataBatchDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more DataBatches.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many DataBatches
     * const dataBatch = await prisma.dataBatch.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends DataBatchUpdateManyArgs>(args: SelectSubset<T, DataBatchUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more DataBatches and returns the data updated in the database.
     * @param {DataBatchUpdateManyAndReturnArgs} args - Arguments to update many DataBatches.
     * @example
     * // Update many DataBatches
     * const dataBatch = await prisma.dataBatch.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more DataBatches and only return the `id`
     * const dataBatchWithIdOnly = await prisma.dataBatch.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends DataBatchUpdateManyAndReturnArgs>(args: SelectSubset<T, DataBatchUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one DataBatch.
     * @param {DataBatchUpsertArgs} args - Arguments to update or create a DataBatch.
     * @example
     * // Update or create a DataBatch
     * const dataBatch = await prisma.dataBatch.upsert({
     *   create: {
     *     // ... data to create a DataBatch
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the DataBatch we want to update
     *   }
     * })
     */
    upsert<T extends DataBatchUpsertArgs>(args: SelectSubset<T, DataBatchUpsertArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of DataBatches.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchCountArgs} args - Arguments to filter DataBatches to count.
     * @example
     * // Count the number of DataBatches
     * const count = await prisma.dataBatch.count({
     *   where: {
     *     // ... the filter for the DataBatches we want to count
     *   }
     * })
    **/
    count<T extends DataBatchCountArgs>(
      args?: Subset<T, DataBatchCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], DataBatchCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a DataBatch.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends DataBatchAggregateArgs>(args: Subset<T, DataBatchAggregateArgs>): Prisma.PrismaPromise<GetDataBatchAggregateType<T>>

    /**
     * Group by DataBatch.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataBatchGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends DataBatchGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: DataBatchGroupByArgs['orderBy'] }
        : { orderBy?: DataBatchGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, DataBatchGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetDataBatchGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the DataBatch model
   */
  readonly fields: DataBatchFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for DataBatch.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__DataBatchClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    samples<T extends DataBatch$samplesArgs<ExtArgs> = {}>(args?: Subset<T, DataBatch$samplesArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the DataBatch model
   */
  interface DataBatchFieldRefs {
    readonly id: FieldRef<"DataBatch", 'Int'>
    readonly name: FieldRef<"DataBatch", 'String'>
    readonly createdAt: FieldRef<"DataBatch", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * DataBatch findUnique
   */
  export type DataBatchFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * Filter, which DataBatch to fetch.
     */
    where: DataBatchWhereUniqueInput
  }

  /**
   * DataBatch findUniqueOrThrow
   */
  export type DataBatchFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * Filter, which DataBatch to fetch.
     */
    where: DataBatchWhereUniqueInput
  }

  /**
   * DataBatch findFirst
   */
  export type DataBatchFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * Filter, which DataBatch to fetch.
     */
    where?: DataBatchWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataBatches to fetch.
     */
    orderBy?: DataBatchOrderByWithRelationInput | DataBatchOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for DataBatches.
     */
    cursor?: DataBatchWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataBatches from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataBatches.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of DataBatches.
     */
    distinct?: DataBatchScalarFieldEnum | DataBatchScalarFieldEnum[]
  }

  /**
   * DataBatch findFirstOrThrow
   */
  export type DataBatchFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * Filter, which DataBatch to fetch.
     */
    where?: DataBatchWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataBatches to fetch.
     */
    orderBy?: DataBatchOrderByWithRelationInput | DataBatchOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for DataBatches.
     */
    cursor?: DataBatchWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataBatches from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataBatches.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of DataBatches.
     */
    distinct?: DataBatchScalarFieldEnum | DataBatchScalarFieldEnum[]
  }

  /**
   * DataBatch findMany
   */
  export type DataBatchFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * Filter, which DataBatches to fetch.
     */
    where?: DataBatchWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataBatches to fetch.
     */
    orderBy?: DataBatchOrderByWithRelationInput | DataBatchOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing DataBatches.
     */
    cursor?: DataBatchWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataBatches from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataBatches.
     */
    skip?: number
    distinct?: DataBatchScalarFieldEnum | DataBatchScalarFieldEnum[]
  }

  /**
   * DataBatch create
   */
  export type DataBatchCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * The data needed to create a DataBatch.
     */
    data: XOR<DataBatchCreateInput, DataBatchUncheckedCreateInput>
  }

  /**
   * DataBatch createMany
   */
  export type DataBatchCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many DataBatches.
     */
    data: DataBatchCreateManyInput | DataBatchCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * DataBatch createManyAndReturn
   */
  export type DataBatchCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * The data used to create many DataBatches.
     */
    data: DataBatchCreateManyInput | DataBatchCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * DataBatch update
   */
  export type DataBatchUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * The data needed to update a DataBatch.
     */
    data: XOR<DataBatchUpdateInput, DataBatchUncheckedUpdateInput>
    /**
     * Choose, which DataBatch to update.
     */
    where: DataBatchWhereUniqueInput
  }

  /**
   * DataBatch updateMany
   */
  export type DataBatchUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update DataBatches.
     */
    data: XOR<DataBatchUpdateManyMutationInput, DataBatchUncheckedUpdateManyInput>
    /**
     * Filter which DataBatches to update
     */
    where?: DataBatchWhereInput
    /**
     * Limit how many DataBatches to update.
     */
    limit?: number
  }

  /**
   * DataBatch updateManyAndReturn
   */
  export type DataBatchUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * The data used to update DataBatches.
     */
    data: XOR<DataBatchUpdateManyMutationInput, DataBatchUncheckedUpdateManyInput>
    /**
     * Filter which DataBatches to update
     */
    where?: DataBatchWhereInput
    /**
     * Limit how many DataBatches to update.
     */
    limit?: number
  }

  /**
   * DataBatch upsert
   */
  export type DataBatchUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * The filter to search for the DataBatch to update in case it exists.
     */
    where: DataBatchWhereUniqueInput
    /**
     * In case the DataBatch found by the `where` argument doesn't exist, create a new DataBatch with this data.
     */
    create: XOR<DataBatchCreateInput, DataBatchUncheckedCreateInput>
    /**
     * In case the DataBatch was found with the provided `where` argument, update it with this data.
     */
    update: XOR<DataBatchUpdateInput, DataBatchUncheckedUpdateInput>
  }

  /**
   * DataBatch delete
   */
  export type DataBatchDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
    /**
     * Filter which DataBatch to delete.
     */
    where: DataBatchWhereUniqueInput
  }

  /**
   * DataBatch deleteMany
   */
  export type DataBatchDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which DataBatches to delete
     */
    where?: DataBatchWhereInput
    /**
     * Limit how many DataBatches to delete.
     */
    limit?: number
  }

  /**
   * DataBatch.samples
   */
  export type DataBatch$samplesArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    where?: DataSampleWhereInput
    orderBy?: DataSampleOrderByWithRelationInput | DataSampleOrderByWithRelationInput[]
    cursor?: DataSampleWhereUniqueInput
    take?: number
    skip?: number
    distinct?: DataSampleScalarFieldEnum | DataSampleScalarFieldEnum[]
  }

  /**
   * DataBatch without action
   */
  export type DataBatchDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataBatch
     */
    select?: DataBatchSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataBatch
     */
    omit?: DataBatchOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataBatchInclude<ExtArgs> | null
  }


  /**
   * Model DataSample
   */

  export type AggregateDataSample = {
    _count: DataSampleCountAggregateOutputType | null
    _avg: DataSampleAvgAggregateOutputType | null
    _sum: DataSampleSumAggregateOutputType | null
    _min: DataSampleMinAggregateOutputType | null
    _max: DataSampleMaxAggregateOutputType | null
  }

  export type DataSampleAvgAggregateOutputType = {
    id: number | null
    batchId: number | null
    luster_value: number | null
    roughness: number | null
    tensile_strength: number | null
  }

  export type DataSampleSumAggregateOutputType = {
    id: number | null
    batchId: number | null
    luster_value: number | null
    roughness: number | null
    tensile_strength: number | null
  }

  export type DataSampleMinAggregateOutputType = {
    id: number | null
    batchId: number | null
    image_capture: string | null
    classification: string | null
    luster_value: number | null
    roughness: number | null
    tensile_strength: number | null
    createdAt: Date | null
  }

  export type DataSampleMaxAggregateOutputType = {
    id: number | null
    batchId: number | null
    image_capture: string | null
    classification: string | null
    luster_value: number | null
    roughness: number | null
    tensile_strength: number | null
    createdAt: Date | null
  }

  export type DataSampleCountAggregateOutputType = {
    id: number
    batchId: number
    image_capture: number
    classification: number
    luster_value: number
    roughness: number
    tensile_strength: number
    createdAt: number
    _all: number
  }


  export type DataSampleAvgAggregateInputType = {
    id?: true
    batchId?: true
    luster_value?: true
    roughness?: true
    tensile_strength?: true
  }

  export type DataSampleSumAggregateInputType = {
    id?: true
    batchId?: true
    luster_value?: true
    roughness?: true
    tensile_strength?: true
  }

  export type DataSampleMinAggregateInputType = {
    id?: true
    batchId?: true
    image_capture?: true
    classification?: true
    luster_value?: true
    roughness?: true
    tensile_strength?: true
    createdAt?: true
  }

  export type DataSampleMaxAggregateInputType = {
    id?: true
    batchId?: true
    image_capture?: true
    classification?: true
    luster_value?: true
    roughness?: true
    tensile_strength?: true
    createdAt?: true
  }

  export type DataSampleCountAggregateInputType = {
    id?: true
    batchId?: true
    image_capture?: true
    classification?: true
    luster_value?: true
    roughness?: true
    tensile_strength?: true
    createdAt?: true
    _all?: true
  }

  export type DataSampleAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which DataSample to aggregate.
     */
    where?: DataSampleWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSamples to fetch.
     */
    orderBy?: DataSampleOrderByWithRelationInput | DataSampleOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: DataSampleWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSamples from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSamples.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned DataSamples
    **/
    _count?: true | DataSampleCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: DataSampleAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: DataSampleSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: DataSampleMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: DataSampleMaxAggregateInputType
  }

  export type GetDataSampleAggregateType<T extends DataSampleAggregateArgs> = {
        [P in keyof T & keyof AggregateDataSample]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateDataSample[P]>
      : GetScalarType<T[P], AggregateDataSample[P]>
  }




  export type DataSampleGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: DataSampleWhereInput
    orderBy?: DataSampleOrderByWithAggregationInput | DataSampleOrderByWithAggregationInput[]
    by: DataSampleScalarFieldEnum[] | DataSampleScalarFieldEnum
    having?: DataSampleScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: DataSampleCountAggregateInputType | true
    _avg?: DataSampleAvgAggregateInputType
    _sum?: DataSampleSumAggregateInputType
    _min?: DataSampleMinAggregateInputType
    _max?: DataSampleMaxAggregateInputType
  }

  export type DataSampleGroupByOutputType = {
    id: number
    batchId: number
    image_capture: string
    classification: string
    luster_value: number | null
    roughness: number | null
    tensile_strength: number | null
    createdAt: Date
    _count: DataSampleCountAggregateOutputType | null
    _avg: DataSampleAvgAggregateOutputType | null
    _sum: DataSampleSumAggregateOutputType | null
    _min: DataSampleMinAggregateOutputType | null
    _max: DataSampleMaxAggregateOutputType | null
  }

  type GetDataSampleGroupByPayload<T extends DataSampleGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<DataSampleGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof DataSampleGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], DataSampleGroupByOutputType[P]>
            : GetScalarType<T[P], DataSampleGroupByOutputType[P]>
        }
      >
    >


  export type DataSampleSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    batchId?: boolean
    image_capture?: boolean
    classification?: boolean
    luster_value?: boolean
    roughness?: boolean
    tensile_strength?: boolean
    createdAt?: boolean
    batch?: boolean | DataBatchDefaultArgs<ExtArgs>
    images?: boolean | DataSample$imagesArgs<ExtArgs>
    _count?: boolean | DataSampleCountOutputTypeDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["dataSample"]>

  export type DataSampleSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    batchId?: boolean
    image_capture?: boolean
    classification?: boolean
    luster_value?: boolean
    roughness?: boolean
    tensile_strength?: boolean
    createdAt?: boolean
    batch?: boolean | DataBatchDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["dataSample"]>

  export type DataSampleSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    batchId?: boolean
    image_capture?: boolean
    classification?: boolean
    luster_value?: boolean
    roughness?: boolean
    tensile_strength?: boolean
    createdAt?: boolean
    batch?: boolean | DataBatchDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["dataSample"]>

  export type DataSampleSelectScalar = {
    id?: boolean
    batchId?: boolean
    image_capture?: boolean
    classification?: boolean
    luster_value?: boolean
    roughness?: boolean
    tensile_strength?: boolean
    createdAt?: boolean
  }

  export type DataSampleOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "batchId" | "image_capture" | "classification" | "luster_value" | "roughness" | "tensile_strength" | "createdAt", ExtArgs["result"]["dataSample"]>
  export type DataSampleInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    batch?: boolean | DataBatchDefaultArgs<ExtArgs>
    images?: boolean | DataSample$imagesArgs<ExtArgs>
    _count?: boolean | DataSampleCountOutputTypeDefaultArgs<ExtArgs>
  }
  export type DataSampleIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    batch?: boolean | DataBatchDefaultArgs<ExtArgs>
  }
  export type DataSampleIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    batch?: boolean | DataBatchDefaultArgs<ExtArgs>
  }

  export type $DataSamplePayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "DataSample"
    objects: {
      batch: Prisma.$DataBatchPayload<ExtArgs>
      images: Prisma.$ImageCapturePayload<ExtArgs>[]
    }
    scalars: $Extensions.GetPayloadResult<{
      id: number
      batchId: number
      image_capture: string
      classification: string
      luster_value: number | null
      roughness: number | null
      tensile_strength: number | null
      createdAt: Date
    }, ExtArgs["result"]["dataSample"]>
    composites: {}
  }

  type DataSampleGetPayload<S extends boolean | null | undefined | DataSampleDefaultArgs> = $Result.GetResult<Prisma.$DataSamplePayload, S>

  type DataSampleCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<DataSampleFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: DataSampleCountAggregateInputType | true
    }

  export interface DataSampleDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['DataSample'], meta: { name: 'DataSample' } }
    /**
     * Find zero or one DataSample that matches the filter.
     * @param {DataSampleFindUniqueArgs} args - Arguments to find a DataSample
     * @example
     * // Get one DataSample
     * const dataSample = await prisma.dataSample.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends DataSampleFindUniqueArgs>(args: SelectSubset<T, DataSampleFindUniqueArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one DataSample that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {DataSampleFindUniqueOrThrowArgs} args - Arguments to find a DataSample
     * @example
     * // Get one DataSample
     * const dataSample = await prisma.dataSample.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends DataSampleFindUniqueOrThrowArgs>(args: SelectSubset<T, DataSampleFindUniqueOrThrowArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first DataSample that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleFindFirstArgs} args - Arguments to find a DataSample
     * @example
     * // Get one DataSample
     * const dataSample = await prisma.dataSample.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends DataSampleFindFirstArgs>(args?: SelectSubset<T, DataSampleFindFirstArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first DataSample that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleFindFirstOrThrowArgs} args - Arguments to find a DataSample
     * @example
     * // Get one DataSample
     * const dataSample = await prisma.dataSample.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends DataSampleFindFirstOrThrowArgs>(args?: SelectSubset<T, DataSampleFindFirstOrThrowArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more DataSamples that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all DataSamples
     * const dataSamples = await prisma.dataSample.findMany()
     * 
     * // Get first 10 DataSamples
     * const dataSamples = await prisma.dataSample.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const dataSampleWithIdOnly = await prisma.dataSample.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends DataSampleFindManyArgs>(args?: SelectSubset<T, DataSampleFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a DataSample.
     * @param {DataSampleCreateArgs} args - Arguments to create a DataSample.
     * @example
     * // Create one DataSample
     * const DataSample = await prisma.dataSample.create({
     *   data: {
     *     // ... data to create a DataSample
     *   }
     * })
     * 
     */
    create<T extends DataSampleCreateArgs>(args: SelectSubset<T, DataSampleCreateArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many DataSamples.
     * @param {DataSampleCreateManyArgs} args - Arguments to create many DataSamples.
     * @example
     * // Create many DataSamples
     * const dataSample = await prisma.dataSample.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends DataSampleCreateManyArgs>(args?: SelectSubset<T, DataSampleCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many DataSamples and returns the data saved in the database.
     * @param {DataSampleCreateManyAndReturnArgs} args - Arguments to create many DataSamples.
     * @example
     * // Create many DataSamples
     * const dataSample = await prisma.dataSample.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many DataSamples and only return the `id`
     * const dataSampleWithIdOnly = await prisma.dataSample.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends DataSampleCreateManyAndReturnArgs>(args?: SelectSubset<T, DataSampleCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a DataSample.
     * @param {DataSampleDeleteArgs} args - Arguments to delete one DataSample.
     * @example
     * // Delete one DataSample
     * const DataSample = await prisma.dataSample.delete({
     *   where: {
     *     // ... filter to delete one DataSample
     *   }
     * })
     * 
     */
    delete<T extends DataSampleDeleteArgs>(args: SelectSubset<T, DataSampleDeleteArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one DataSample.
     * @param {DataSampleUpdateArgs} args - Arguments to update one DataSample.
     * @example
     * // Update one DataSample
     * const dataSample = await prisma.dataSample.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends DataSampleUpdateArgs>(args: SelectSubset<T, DataSampleUpdateArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more DataSamples.
     * @param {DataSampleDeleteManyArgs} args - Arguments to filter DataSamples to delete.
     * @example
     * // Delete a few DataSamples
     * const { count } = await prisma.dataSample.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends DataSampleDeleteManyArgs>(args?: SelectSubset<T, DataSampleDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more DataSamples.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many DataSamples
     * const dataSample = await prisma.dataSample.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends DataSampleUpdateManyArgs>(args: SelectSubset<T, DataSampleUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more DataSamples and returns the data updated in the database.
     * @param {DataSampleUpdateManyAndReturnArgs} args - Arguments to update many DataSamples.
     * @example
     * // Update many DataSamples
     * const dataSample = await prisma.dataSample.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more DataSamples and only return the `id`
     * const dataSampleWithIdOnly = await prisma.dataSample.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends DataSampleUpdateManyAndReturnArgs>(args: SelectSubset<T, DataSampleUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one DataSample.
     * @param {DataSampleUpsertArgs} args - Arguments to update or create a DataSample.
     * @example
     * // Update or create a DataSample
     * const dataSample = await prisma.dataSample.upsert({
     *   create: {
     *     // ... data to create a DataSample
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the DataSample we want to update
     *   }
     * })
     */
    upsert<T extends DataSampleUpsertArgs>(args: SelectSubset<T, DataSampleUpsertArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of DataSamples.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleCountArgs} args - Arguments to filter DataSamples to count.
     * @example
     * // Count the number of DataSamples
     * const count = await prisma.dataSample.count({
     *   where: {
     *     // ... the filter for the DataSamples we want to count
     *   }
     * })
    **/
    count<T extends DataSampleCountArgs>(
      args?: Subset<T, DataSampleCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], DataSampleCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a DataSample.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends DataSampleAggregateArgs>(args: Subset<T, DataSampleAggregateArgs>): Prisma.PrismaPromise<GetDataSampleAggregateType<T>>

    /**
     * Group by DataSample.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {DataSampleGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends DataSampleGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: DataSampleGroupByArgs['orderBy'] }
        : { orderBy?: DataSampleGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, DataSampleGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetDataSampleGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the DataSample model
   */
  readonly fields: DataSampleFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for DataSample.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__DataSampleClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    batch<T extends DataBatchDefaultArgs<ExtArgs> = {}>(args?: Subset<T, DataBatchDefaultArgs<ExtArgs>>): Prisma__DataBatchClient<$Result.GetResult<Prisma.$DataBatchPayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    images<T extends DataSample$imagesArgs<ExtArgs> = {}>(args?: Subset<T, DataSample$imagesArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "findMany", GlobalOmitOptions> | Null>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the DataSample model
   */
  interface DataSampleFieldRefs {
    readonly id: FieldRef<"DataSample", 'Int'>
    readonly batchId: FieldRef<"DataSample", 'Int'>
    readonly image_capture: FieldRef<"DataSample", 'String'>
    readonly classification: FieldRef<"DataSample", 'String'>
    readonly luster_value: FieldRef<"DataSample", 'Float'>
    readonly roughness: FieldRef<"DataSample", 'Float'>
    readonly tensile_strength: FieldRef<"DataSample", 'Float'>
    readonly createdAt: FieldRef<"DataSample", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * DataSample findUnique
   */
  export type DataSampleFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * Filter, which DataSample to fetch.
     */
    where: DataSampleWhereUniqueInput
  }

  /**
   * DataSample findUniqueOrThrow
   */
  export type DataSampleFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * Filter, which DataSample to fetch.
     */
    where: DataSampleWhereUniqueInput
  }

  /**
   * DataSample findFirst
   */
  export type DataSampleFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * Filter, which DataSample to fetch.
     */
    where?: DataSampleWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSamples to fetch.
     */
    orderBy?: DataSampleOrderByWithRelationInput | DataSampleOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for DataSamples.
     */
    cursor?: DataSampleWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSamples from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSamples.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of DataSamples.
     */
    distinct?: DataSampleScalarFieldEnum | DataSampleScalarFieldEnum[]
  }

  /**
   * DataSample findFirstOrThrow
   */
  export type DataSampleFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * Filter, which DataSample to fetch.
     */
    where?: DataSampleWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSamples to fetch.
     */
    orderBy?: DataSampleOrderByWithRelationInput | DataSampleOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for DataSamples.
     */
    cursor?: DataSampleWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSamples from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSamples.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of DataSamples.
     */
    distinct?: DataSampleScalarFieldEnum | DataSampleScalarFieldEnum[]
  }

  /**
   * DataSample findMany
   */
  export type DataSampleFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * Filter, which DataSamples to fetch.
     */
    where?: DataSampleWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of DataSamples to fetch.
     */
    orderBy?: DataSampleOrderByWithRelationInput | DataSampleOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing DataSamples.
     */
    cursor?: DataSampleWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` DataSamples from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` DataSamples.
     */
    skip?: number
    distinct?: DataSampleScalarFieldEnum | DataSampleScalarFieldEnum[]
  }

  /**
   * DataSample create
   */
  export type DataSampleCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * The data needed to create a DataSample.
     */
    data: XOR<DataSampleCreateInput, DataSampleUncheckedCreateInput>
  }

  /**
   * DataSample createMany
   */
  export type DataSampleCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many DataSamples.
     */
    data: DataSampleCreateManyInput | DataSampleCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * DataSample createManyAndReturn
   */
  export type DataSampleCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * The data used to create many DataSamples.
     */
    data: DataSampleCreateManyInput | DataSampleCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * DataSample update
   */
  export type DataSampleUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * The data needed to update a DataSample.
     */
    data: XOR<DataSampleUpdateInput, DataSampleUncheckedUpdateInput>
    /**
     * Choose, which DataSample to update.
     */
    where: DataSampleWhereUniqueInput
  }

  /**
   * DataSample updateMany
   */
  export type DataSampleUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update DataSamples.
     */
    data: XOR<DataSampleUpdateManyMutationInput, DataSampleUncheckedUpdateManyInput>
    /**
     * Filter which DataSamples to update
     */
    where?: DataSampleWhereInput
    /**
     * Limit how many DataSamples to update.
     */
    limit?: number
  }

  /**
   * DataSample updateManyAndReturn
   */
  export type DataSampleUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * The data used to update DataSamples.
     */
    data: XOR<DataSampleUpdateManyMutationInput, DataSampleUncheckedUpdateManyInput>
    /**
     * Filter which DataSamples to update
     */
    where?: DataSampleWhereInput
    /**
     * Limit how many DataSamples to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * DataSample upsert
   */
  export type DataSampleUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * The filter to search for the DataSample to update in case it exists.
     */
    where: DataSampleWhereUniqueInput
    /**
     * In case the DataSample found by the `where` argument doesn't exist, create a new DataSample with this data.
     */
    create: XOR<DataSampleCreateInput, DataSampleUncheckedCreateInput>
    /**
     * In case the DataSample was found with the provided `where` argument, update it with this data.
     */
    update: XOR<DataSampleUpdateInput, DataSampleUncheckedUpdateInput>
  }

  /**
   * DataSample delete
   */
  export type DataSampleDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
    /**
     * Filter which DataSample to delete.
     */
    where: DataSampleWhereUniqueInput
  }

  /**
   * DataSample deleteMany
   */
  export type DataSampleDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which DataSamples to delete
     */
    where?: DataSampleWhereInput
    /**
     * Limit how many DataSamples to delete.
     */
    limit?: number
  }

  /**
   * DataSample.images
   */
  export type DataSample$imagesArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    where?: ImageCaptureWhereInput
    orderBy?: ImageCaptureOrderByWithRelationInput | ImageCaptureOrderByWithRelationInput[]
    cursor?: ImageCaptureWhereUniqueInput
    take?: number
    skip?: number
    distinct?: ImageCaptureScalarFieldEnum | ImageCaptureScalarFieldEnum[]
  }

  /**
   * DataSample without action
   */
  export type DataSampleDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the DataSample
     */
    select?: DataSampleSelect<ExtArgs> | null
    /**
     * Omit specific fields from the DataSample
     */
    omit?: DataSampleOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: DataSampleInclude<ExtArgs> | null
  }


  /**
   * Model ImageCapture
   */

  export type AggregateImageCapture = {
    _count: ImageCaptureCountAggregateOutputType | null
    _avg: ImageCaptureAvgAggregateOutputType | null
    _sum: ImageCaptureSumAggregateOutputType | null
    _min: ImageCaptureMinAggregateOutputType | null
    _max: ImageCaptureMaxAggregateOutputType | null
  }

  export type ImageCaptureAvgAggregateOutputType = {
    id: number | null
    sampleId: number | null
  }

  export type ImageCaptureSumAggregateOutputType = {
    id: number | null
    sampleId: number | null
  }

  export type ImageCaptureMinAggregateOutputType = {
    id: number | null
    sampleId: number | null
    fileName: string | null
    imageUrl: string | null
    createdAt: Date | null
  }

  export type ImageCaptureMaxAggregateOutputType = {
    id: number | null
    sampleId: number | null
    fileName: string | null
    imageUrl: string | null
    createdAt: Date | null
  }

  export type ImageCaptureCountAggregateOutputType = {
    id: number
    sampleId: number
    fileName: number
    imageUrl: number
    createdAt: number
    _all: number
  }


  export type ImageCaptureAvgAggregateInputType = {
    id?: true
    sampleId?: true
  }

  export type ImageCaptureSumAggregateInputType = {
    id?: true
    sampleId?: true
  }

  export type ImageCaptureMinAggregateInputType = {
    id?: true
    sampleId?: true
    fileName?: true
    imageUrl?: true
    createdAt?: true
  }

  export type ImageCaptureMaxAggregateInputType = {
    id?: true
    sampleId?: true
    fileName?: true
    imageUrl?: true
    createdAt?: true
  }

  export type ImageCaptureCountAggregateInputType = {
    id?: true
    sampleId?: true
    fileName?: true
    imageUrl?: true
    createdAt?: true
    _all?: true
  }

  export type ImageCaptureAggregateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which ImageCapture to aggregate.
     */
    where?: ImageCaptureWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ImageCaptures to fetch.
     */
    orderBy?: ImageCaptureOrderByWithRelationInput | ImageCaptureOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the start position
     */
    cursor?: ImageCaptureWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ImageCaptures from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ImageCaptures.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Count returned ImageCaptures
    **/
    _count?: true | ImageCaptureCountAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to average
    **/
    _avg?: ImageCaptureAvgAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to sum
    **/
    _sum?: ImageCaptureSumAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the minimum value
    **/
    _min?: ImageCaptureMinAggregateInputType
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/aggregations Aggregation Docs}
     * 
     * Select which fields to find the maximum value
    **/
    _max?: ImageCaptureMaxAggregateInputType
  }

  export type GetImageCaptureAggregateType<T extends ImageCaptureAggregateArgs> = {
        [P in keyof T & keyof AggregateImageCapture]: P extends '_count' | 'count'
      ? T[P] extends true
        ? number
        : GetScalarType<T[P], AggregateImageCapture[P]>
      : GetScalarType<T[P], AggregateImageCapture[P]>
  }




  export type ImageCaptureGroupByArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    where?: ImageCaptureWhereInput
    orderBy?: ImageCaptureOrderByWithAggregationInput | ImageCaptureOrderByWithAggregationInput[]
    by: ImageCaptureScalarFieldEnum[] | ImageCaptureScalarFieldEnum
    having?: ImageCaptureScalarWhereWithAggregatesInput
    take?: number
    skip?: number
    _count?: ImageCaptureCountAggregateInputType | true
    _avg?: ImageCaptureAvgAggregateInputType
    _sum?: ImageCaptureSumAggregateInputType
    _min?: ImageCaptureMinAggregateInputType
    _max?: ImageCaptureMaxAggregateInputType
  }

  export type ImageCaptureGroupByOutputType = {
    id: number
    sampleId: number
    fileName: string | null
    imageUrl: string
    createdAt: Date
    _count: ImageCaptureCountAggregateOutputType | null
    _avg: ImageCaptureAvgAggregateOutputType | null
    _sum: ImageCaptureSumAggregateOutputType | null
    _min: ImageCaptureMinAggregateOutputType | null
    _max: ImageCaptureMaxAggregateOutputType | null
  }

  type GetImageCaptureGroupByPayload<T extends ImageCaptureGroupByArgs> = Prisma.PrismaPromise<
    Array<
      PickEnumerable<ImageCaptureGroupByOutputType, T['by']> &
        {
          [P in ((keyof T) & (keyof ImageCaptureGroupByOutputType))]: P extends '_count'
            ? T[P] extends boolean
              ? number
              : GetScalarType<T[P], ImageCaptureGroupByOutputType[P]>
            : GetScalarType<T[P], ImageCaptureGroupByOutputType[P]>
        }
      >
    >


  export type ImageCaptureSelect<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    sampleId?: boolean
    fileName?: boolean
    imageUrl?: boolean
    createdAt?: boolean
    sample?: boolean | DataSampleDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["imageCapture"]>

  export type ImageCaptureSelectCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    sampleId?: boolean
    fileName?: boolean
    imageUrl?: boolean
    createdAt?: boolean
    sample?: boolean | DataSampleDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["imageCapture"]>

  export type ImageCaptureSelectUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetSelect<{
    id?: boolean
    sampleId?: boolean
    fileName?: boolean
    imageUrl?: boolean
    createdAt?: boolean
    sample?: boolean | DataSampleDefaultArgs<ExtArgs>
  }, ExtArgs["result"]["imageCapture"]>

  export type ImageCaptureSelectScalar = {
    id?: boolean
    sampleId?: boolean
    fileName?: boolean
    imageUrl?: boolean
    createdAt?: boolean
  }

  export type ImageCaptureOmit<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = $Extensions.GetOmit<"id" | "sampleId" | "fileName" | "imageUrl" | "createdAt", ExtArgs["result"]["imageCapture"]>
  export type ImageCaptureInclude<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    sample?: boolean | DataSampleDefaultArgs<ExtArgs>
  }
  export type ImageCaptureIncludeCreateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    sample?: boolean | DataSampleDefaultArgs<ExtArgs>
  }
  export type ImageCaptureIncludeUpdateManyAndReturn<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    sample?: boolean | DataSampleDefaultArgs<ExtArgs>
  }

  export type $ImageCapturePayload<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    name: "ImageCapture"
    objects: {
      sample: Prisma.$DataSamplePayload<ExtArgs>
    }
    scalars: $Extensions.GetPayloadResult<{
      id: number
      sampleId: number
      fileName: string | null
      imageUrl: string
      createdAt: Date
    }, ExtArgs["result"]["imageCapture"]>
    composites: {}
  }

  type ImageCaptureGetPayload<S extends boolean | null | undefined | ImageCaptureDefaultArgs> = $Result.GetResult<Prisma.$ImageCapturePayload, S>

  type ImageCaptureCountArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> =
    Omit<ImageCaptureFindManyArgs, 'select' | 'include' | 'distinct' | 'omit'> & {
      select?: ImageCaptureCountAggregateInputType | true
    }

  export interface ImageCaptureDelegate<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> {
    [K: symbol]: { types: Prisma.TypeMap<ExtArgs>['model']['ImageCapture'], meta: { name: 'ImageCapture' } }
    /**
     * Find zero or one ImageCapture that matches the filter.
     * @param {ImageCaptureFindUniqueArgs} args - Arguments to find a ImageCapture
     * @example
     * // Get one ImageCapture
     * const imageCapture = await prisma.imageCapture.findUnique({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUnique<T extends ImageCaptureFindUniqueArgs>(args: SelectSubset<T, ImageCaptureFindUniqueArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "findUnique", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find one ImageCapture that matches the filter or throw an error with `error.code='P2025'`
     * if no matches were found.
     * @param {ImageCaptureFindUniqueOrThrowArgs} args - Arguments to find a ImageCapture
     * @example
     * // Get one ImageCapture
     * const imageCapture = await prisma.imageCapture.findUniqueOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findUniqueOrThrow<T extends ImageCaptureFindUniqueOrThrowArgs>(args: SelectSubset<T, ImageCaptureFindUniqueOrThrowArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first ImageCapture that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureFindFirstArgs} args - Arguments to find a ImageCapture
     * @example
     * // Get one ImageCapture
     * const imageCapture = await prisma.imageCapture.findFirst({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirst<T extends ImageCaptureFindFirstArgs>(args?: SelectSubset<T, ImageCaptureFindFirstArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "findFirst", GlobalOmitOptions> | null, null, ExtArgs, GlobalOmitOptions>

    /**
     * Find the first ImageCapture that matches the filter or
     * throw `PrismaKnownClientError` with `P2025` code if no matches were found.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureFindFirstOrThrowArgs} args - Arguments to find a ImageCapture
     * @example
     * // Get one ImageCapture
     * const imageCapture = await prisma.imageCapture.findFirstOrThrow({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     */
    findFirstOrThrow<T extends ImageCaptureFindFirstOrThrowArgs>(args?: SelectSubset<T, ImageCaptureFindFirstOrThrowArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "findFirstOrThrow", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Find zero or more ImageCaptures that matches the filter.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureFindManyArgs} args - Arguments to filter and select certain fields only.
     * @example
     * // Get all ImageCaptures
     * const imageCaptures = await prisma.imageCapture.findMany()
     * 
     * // Get first 10 ImageCaptures
     * const imageCaptures = await prisma.imageCapture.findMany({ take: 10 })
     * 
     * // Only select the `id`
     * const imageCaptureWithIdOnly = await prisma.imageCapture.findMany({ select: { id: true } })
     * 
     */
    findMany<T extends ImageCaptureFindManyArgs>(args?: SelectSubset<T, ImageCaptureFindManyArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "findMany", GlobalOmitOptions>>

    /**
     * Create a ImageCapture.
     * @param {ImageCaptureCreateArgs} args - Arguments to create a ImageCapture.
     * @example
     * // Create one ImageCapture
     * const ImageCapture = await prisma.imageCapture.create({
     *   data: {
     *     // ... data to create a ImageCapture
     *   }
     * })
     * 
     */
    create<T extends ImageCaptureCreateArgs>(args: SelectSubset<T, ImageCaptureCreateArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "create", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Create many ImageCaptures.
     * @param {ImageCaptureCreateManyArgs} args - Arguments to create many ImageCaptures.
     * @example
     * // Create many ImageCaptures
     * const imageCapture = await prisma.imageCapture.createMany({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     *     
     */
    createMany<T extends ImageCaptureCreateManyArgs>(args?: SelectSubset<T, ImageCaptureCreateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Create many ImageCaptures and returns the data saved in the database.
     * @param {ImageCaptureCreateManyAndReturnArgs} args - Arguments to create many ImageCaptures.
     * @example
     * // Create many ImageCaptures
     * const imageCapture = await prisma.imageCapture.createManyAndReturn({
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Create many ImageCaptures and only return the `id`
     * const imageCaptureWithIdOnly = await prisma.imageCapture.createManyAndReturn({
     *   select: { id: true },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    createManyAndReturn<T extends ImageCaptureCreateManyAndReturnArgs>(args?: SelectSubset<T, ImageCaptureCreateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "createManyAndReturn", GlobalOmitOptions>>

    /**
     * Delete a ImageCapture.
     * @param {ImageCaptureDeleteArgs} args - Arguments to delete one ImageCapture.
     * @example
     * // Delete one ImageCapture
     * const ImageCapture = await prisma.imageCapture.delete({
     *   where: {
     *     // ... filter to delete one ImageCapture
     *   }
     * })
     * 
     */
    delete<T extends ImageCaptureDeleteArgs>(args: SelectSubset<T, ImageCaptureDeleteArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "delete", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Update one ImageCapture.
     * @param {ImageCaptureUpdateArgs} args - Arguments to update one ImageCapture.
     * @example
     * // Update one ImageCapture
     * const imageCapture = await prisma.imageCapture.update({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    update<T extends ImageCaptureUpdateArgs>(args: SelectSubset<T, ImageCaptureUpdateArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "update", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>

    /**
     * Delete zero or more ImageCaptures.
     * @param {ImageCaptureDeleteManyArgs} args - Arguments to filter ImageCaptures to delete.
     * @example
     * // Delete a few ImageCaptures
     * const { count } = await prisma.imageCapture.deleteMany({
     *   where: {
     *     // ... provide filter here
     *   }
     * })
     * 
     */
    deleteMany<T extends ImageCaptureDeleteManyArgs>(args?: SelectSubset<T, ImageCaptureDeleteManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more ImageCaptures.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureUpdateManyArgs} args - Arguments to update one or more rows.
     * @example
     * // Update many ImageCaptures
     * const imageCapture = await prisma.imageCapture.updateMany({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: {
     *     // ... provide data here
     *   }
     * })
     * 
     */
    updateMany<T extends ImageCaptureUpdateManyArgs>(args: SelectSubset<T, ImageCaptureUpdateManyArgs<ExtArgs>>): Prisma.PrismaPromise<BatchPayload>

    /**
     * Update zero or more ImageCaptures and returns the data updated in the database.
     * @param {ImageCaptureUpdateManyAndReturnArgs} args - Arguments to update many ImageCaptures.
     * @example
     * // Update many ImageCaptures
     * const imageCapture = await prisma.imageCapture.updateManyAndReturn({
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * 
     * // Update zero or more ImageCaptures and only return the `id`
     * const imageCaptureWithIdOnly = await prisma.imageCapture.updateManyAndReturn({
     *   select: { id: true },
     *   where: {
     *     // ... provide filter here
     *   },
     *   data: [
     *     // ... provide data here
     *   ]
     * })
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * 
     */
    updateManyAndReturn<T extends ImageCaptureUpdateManyAndReturnArgs>(args: SelectSubset<T, ImageCaptureUpdateManyAndReturnArgs<ExtArgs>>): Prisma.PrismaPromise<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "updateManyAndReturn", GlobalOmitOptions>>

    /**
     * Create or update one ImageCapture.
     * @param {ImageCaptureUpsertArgs} args - Arguments to update or create a ImageCapture.
     * @example
     * // Update or create a ImageCapture
     * const imageCapture = await prisma.imageCapture.upsert({
     *   create: {
     *     // ... data to create a ImageCapture
     *   },
     *   update: {
     *     // ... in case it already exists, update
     *   },
     *   where: {
     *     // ... the filter for the ImageCapture we want to update
     *   }
     * })
     */
    upsert<T extends ImageCaptureUpsertArgs>(args: SelectSubset<T, ImageCaptureUpsertArgs<ExtArgs>>): Prisma__ImageCaptureClient<$Result.GetResult<Prisma.$ImageCapturePayload<ExtArgs>, T, "upsert", GlobalOmitOptions>, never, ExtArgs, GlobalOmitOptions>


    /**
     * Count the number of ImageCaptures.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureCountArgs} args - Arguments to filter ImageCaptures to count.
     * @example
     * // Count the number of ImageCaptures
     * const count = await prisma.imageCapture.count({
     *   where: {
     *     // ... the filter for the ImageCaptures we want to count
     *   }
     * })
    **/
    count<T extends ImageCaptureCountArgs>(
      args?: Subset<T, ImageCaptureCountArgs>,
    ): Prisma.PrismaPromise<
      T extends $Utils.Record<'select', any>
        ? T['select'] extends true
          ? number
          : GetScalarType<T['select'], ImageCaptureCountAggregateOutputType>
        : number
    >

    /**
     * Allows you to perform aggregations operations on a ImageCapture.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureAggregateArgs} args - Select which aggregations you would like to apply and on what fields.
     * @example
     * // Ordered by age ascending
     * // Where email contains prisma.io
     * // Limited to the 10 users
     * const aggregations = await prisma.user.aggregate({
     *   _avg: {
     *     age: true,
     *   },
     *   where: {
     *     email: {
     *       contains: "prisma.io",
     *     },
     *   },
     *   orderBy: {
     *     age: "asc",
     *   },
     *   take: 10,
     * })
    **/
    aggregate<T extends ImageCaptureAggregateArgs>(args: Subset<T, ImageCaptureAggregateArgs>): Prisma.PrismaPromise<GetImageCaptureAggregateType<T>>

    /**
     * Group by ImageCapture.
     * Note, that providing `undefined` is treated as the value not being there.
     * Read more here: https://pris.ly/d/null-undefined
     * @param {ImageCaptureGroupByArgs} args - Group by arguments.
     * @example
     * // Group by city, order by createdAt, get count
     * const result = await prisma.user.groupBy({
     *   by: ['city', 'createdAt'],
     *   orderBy: {
     *     createdAt: true
     *   },
     *   _count: {
     *     _all: true
     *   },
     * })
     * 
    **/
    groupBy<
      T extends ImageCaptureGroupByArgs,
      HasSelectOrTake extends Or<
        Extends<'skip', Keys<T>>,
        Extends<'take', Keys<T>>
      >,
      OrderByArg extends True extends HasSelectOrTake
        ? { orderBy: ImageCaptureGroupByArgs['orderBy'] }
        : { orderBy?: ImageCaptureGroupByArgs['orderBy'] },
      OrderFields extends ExcludeUnderscoreKeys<Keys<MaybeTupleToUnion<T['orderBy']>>>,
      ByFields extends MaybeTupleToUnion<T['by']>,
      ByValid extends Has<ByFields, OrderFields>,
      HavingFields extends GetHavingFields<T['having']>,
      HavingValid extends Has<ByFields, HavingFields>,
      ByEmpty extends T['by'] extends never[] ? True : False,
      InputErrors extends ByEmpty extends True
      ? `Error: "by" must not be empty.`
      : HavingValid extends False
      ? {
          [P in HavingFields]: P extends ByFields
            ? never
            : P extends string
            ? `Error: Field "${P}" used in "having" needs to be provided in "by".`
            : [
                Error,
                'Field ',
                P,
                ` in "having" needs to be provided in "by"`,
              ]
        }[HavingFields]
      : 'take' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "take", you also need to provide "orderBy"'
      : 'skip' extends Keys<T>
      ? 'orderBy' extends Keys<T>
        ? ByValid extends True
          ? {}
          : {
              [P in OrderFields]: P extends ByFields
                ? never
                : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
            }[OrderFields]
        : 'Error: If you provide "skip", you also need to provide "orderBy"'
      : ByValid extends True
      ? {}
      : {
          [P in OrderFields]: P extends ByFields
            ? never
            : `Error: Field "${P}" in "orderBy" needs to be provided in "by"`
        }[OrderFields]
    >(args: SubsetIntersection<T, ImageCaptureGroupByArgs, OrderByArg> & InputErrors): {} extends InputErrors ? GetImageCaptureGroupByPayload<T> : Prisma.PrismaPromise<InputErrors>
  /**
   * Fields of the ImageCapture model
   */
  readonly fields: ImageCaptureFieldRefs;
  }

  /**
   * The delegate class that acts as a "Promise-like" for ImageCapture.
   * Why is this prefixed with `Prisma__`?
   * Because we want to prevent naming conflicts as mentioned in
   * https://github.com/prisma/prisma-client-js/issues/707
   */
  export interface Prisma__ImageCaptureClient<T, Null = never, ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs, GlobalOmitOptions = {}> extends Prisma.PrismaPromise<T> {
    readonly [Symbol.toStringTag]: "PrismaPromise"
    sample<T extends DataSampleDefaultArgs<ExtArgs> = {}>(args?: Subset<T, DataSampleDefaultArgs<ExtArgs>>): Prisma__DataSampleClient<$Result.GetResult<Prisma.$DataSamplePayload<ExtArgs>, T, "findUniqueOrThrow", GlobalOmitOptions> | Null, Null, ExtArgs, GlobalOmitOptions>
    /**
     * Attaches callbacks for the resolution and/or rejection of the Promise.
     * @param onfulfilled The callback to execute when the Promise is resolved.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of which ever callback is executed.
     */
    then<TResult1 = T, TResult2 = never>(onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | undefined | null, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | undefined | null): $Utils.JsPromise<TResult1 | TResult2>
    /**
     * Attaches a callback for only the rejection of the Promise.
     * @param onrejected The callback to execute when the Promise is rejected.
     * @returns A Promise for the completion of the callback.
     */
    catch<TResult = never>(onrejected?: ((reason: any) => TResult | PromiseLike<TResult>) | undefined | null): $Utils.JsPromise<T | TResult>
    /**
     * Attaches a callback that is invoked when the Promise is settled (fulfilled or rejected). The
     * resolved value cannot be modified from the callback.
     * @param onfinally The callback to execute when the Promise is settled (fulfilled or rejected).
     * @returns A Promise for the completion of the callback.
     */
    finally(onfinally?: (() => void) | undefined | null): $Utils.JsPromise<T>
  }




  /**
   * Fields of the ImageCapture model
   */
  interface ImageCaptureFieldRefs {
    readonly id: FieldRef<"ImageCapture", 'Int'>
    readonly sampleId: FieldRef<"ImageCapture", 'Int'>
    readonly fileName: FieldRef<"ImageCapture", 'String'>
    readonly imageUrl: FieldRef<"ImageCapture", 'String'>
    readonly createdAt: FieldRef<"ImageCapture", 'DateTime'>
  }
    

  // Custom InputTypes
  /**
   * ImageCapture findUnique
   */
  export type ImageCaptureFindUniqueArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * Filter, which ImageCapture to fetch.
     */
    where: ImageCaptureWhereUniqueInput
  }

  /**
   * ImageCapture findUniqueOrThrow
   */
  export type ImageCaptureFindUniqueOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * Filter, which ImageCapture to fetch.
     */
    where: ImageCaptureWhereUniqueInput
  }

  /**
   * ImageCapture findFirst
   */
  export type ImageCaptureFindFirstArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * Filter, which ImageCapture to fetch.
     */
    where?: ImageCaptureWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ImageCaptures to fetch.
     */
    orderBy?: ImageCaptureOrderByWithRelationInput | ImageCaptureOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for ImageCaptures.
     */
    cursor?: ImageCaptureWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ImageCaptures from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ImageCaptures.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of ImageCaptures.
     */
    distinct?: ImageCaptureScalarFieldEnum | ImageCaptureScalarFieldEnum[]
  }

  /**
   * ImageCapture findFirstOrThrow
   */
  export type ImageCaptureFindFirstOrThrowArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * Filter, which ImageCapture to fetch.
     */
    where?: ImageCaptureWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ImageCaptures to fetch.
     */
    orderBy?: ImageCaptureOrderByWithRelationInput | ImageCaptureOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for searching for ImageCaptures.
     */
    cursor?: ImageCaptureWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ImageCaptures from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ImageCaptures.
     */
    skip?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/distinct Distinct Docs}
     * 
     * Filter by unique combinations of ImageCaptures.
     */
    distinct?: ImageCaptureScalarFieldEnum | ImageCaptureScalarFieldEnum[]
  }

  /**
   * ImageCapture findMany
   */
  export type ImageCaptureFindManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * Filter, which ImageCaptures to fetch.
     */
    where?: ImageCaptureWhereInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/sorting Sorting Docs}
     * 
     * Determine the order of ImageCaptures to fetch.
     */
    orderBy?: ImageCaptureOrderByWithRelationInput | ImageCaptureOrderByWithRelationInput[]
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination#cursor-based-pagination Cursor Docs}
     * 
     * Sets the position for listing ImageCaptures.
     */
    cursor?: ImageCaptureWhereUniqueInput
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Take `±n` ImageCaptures from the position of the cursor.
     */
    take?: number
    /**
     * {@link https://www.prisma.io/docs/concepts/components/prisma-client/pagination Pagination Docs}
     * 
     * Skip the first `n` ImageCaptures.
     */
    skip?: number
    distinct?: ImageCaptureScalarFieldEnum | ImageCaptureScalarFieldEnum[]
  }

  /**
   * ImageCapture create
   */
  export type ImageCaptureCreateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * The data needed to create a ImageCapture.
     */
    data: XOR<ImageCaptureCreateInput, ImageCaptureUncheckedCreateInput>
  }

  /**
   * ImageCapture createMany
   */
  export type ImageCaptureCreateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to create many ImageCaptures.
     */
    data: ImageCaptureCreateManyInput | ImageCaptureCreateManyInput[]
    skipDuplicates?: boolean
  }

  /**
   * ImageCapture createManyAndReturn
   */
  export type ImageCaptureCreateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelectCreateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * The data used to create many ImageCaptures.
     */
    data: ImageCaptureCreateManyInput | ImageCaptureCreateManyInput[]
    skipDuplicates?: boolean
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureIncludeCreateManyAndReturn<ExtArgs> | null
  }

  /**
   * ImageCapture update
   */
  export type ImageCaptureUpdateArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * The data needed to update a ImageCapture.
     */
    data: XOR<ImageCaptureUpdateInput, ImageCaptureUncheckedUpdateInput>
    /**
     * Choose, which ImageCapture to update.
     */
    where: ImageCaptureWhereUniqueInput
  }

  /**
   * ImageCapture updateMany
   */
  export type ImageCaptureUpdateManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * The data used to update ImageCaptures.
     */
    data: XOR<ImageCaptureUpdateManyMutationInput, ImageCaptureUncheckedUpdateManyInput>
    /**
     * Filter which ImageCaptures to update
     */
    where?: ImageCaptureWhereInput
    /**
     * Limit how many ImageCaptures to update.
     */
    limit?: number
  }

  /**
   * ImageCapture updateManyAndReturn
   */
  export type ImageCaptureUpdateManyAndReturnArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelectUpdateManyAndReturn<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * The data used to update ImageCaptures.
     */
    data: XOR<ImageCaptureUpdateManyMutationInput, ImageCaptureUncheckedUpdateManyInput>
    /**
     * Filter which ImageCaptures to update
     */
    where?: ImageCaptureWhereInput
    /**
     * Limit how many ImageCaptures to update.
     */
    limit?: number
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureIncludeUpdateManyAndReturn<ExtArgs> | null
  }

  /**
   * ImageCapture upsert
   */
  export type ImageCaptureUpsertArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * The filter to search for the ImageCapture to update in case it exists.
     */
    where: ImageCaptureWhereUniqueInput
    /**
     * In case the ImageCapture found by the `where` argument doesn't exist, create a new ImageCapture with this data.
     */
    create: XOR<ImageCaptureCreateInput, ImageCaptureUncheckedCreateInput>
    /**
     * In case the ImageCapture was found with the provided `where` argument, update it with this data.
     */
    update: XOR<ImageCaptureUpdateInput, ImageCaptureUncheckedUpdateInput>
  }

  /**
   * ImageCapture delete
   */
  export type ImageCaptureDeleteArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
    /**
     * Filter which ImageCapture to delete.
     */
    where: ImageCaptureWhereUniqueInput
  }

  /**
   * ImageCapture deleteMany
   */
  export type ImageCaptureDeleteManyArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Filter which ImageCaptures to delete
     */
    where?: ImageCaptureWhereInput
    /**
     * Limit how many ImageCaptures to delete.
     */
    limit?: number
  }

  /**
   * ImageCapture without action
   */
  export type ImageCaptureDefaultArgs<ExtArgs extends $Extensions.InternalArgs = $Extensions.DefaultArgs> = {
    /**
     * Select specific fields to fetch from the ImageCapture
     */
    select?: ImageCaptureSelect<ExtArgs> | null
    /**
     * Omit specific fields from the ImageCapture
     */
    omit?: ImageCaptureOmit<ExtArgs> | null
    /**
     * Choose, which related nodes to fetch as well
     */
    include?: ImageCaptureInclude<ExtArgs> | null
  }


  /**
   * Enums
   */

  export const TransactionIsolationLevel: {
    ReadUncommitted: 'ReadUncommitted',
    ReadCommitted: 'ReadCommitted',
    RepeatableRead: 'RepeatableRead',
    Serializable: 'Serializable'
  };

  export type TransactionIsolationLevel = (typeof TransactionIsolationLevel)[keyof typeof TransactionIsolationLevel]


  export const UserScalarFieldEnum: {
    id: 'id',
    email: 'email',
    password: 'password',
    name: 'name',
    createdAt: 'createdAt'
  };

  export type UserScalarFieldEnum = (typeof UserScalarFieldEnum)[keyof typeof UserScalarFieldEnum]


  export const DataBatchScalarFieldEnum: {
    id: 'id',
    name: 'name',
    createdAt: 'createdAt'
  };

  export type DataBatchScalarFieldEnum = (typeof DataBatchScalarFieldEnum)[keyof typeof DataBatchScalarFieldEnum]


  export const DataSampleScalarFieldEnum: {
    id: 'id',
    batchId: 'batchId',
    image_capture: 'image_capture',
    classification: 'classification',
    luster_value: 'luster_value',
    roughness: 'roughness',
    tensile_strength: 'tensile_strength',
    createdAt: 'createdAt'
  };

  export type DataSampleScalarFieldEnum = (typeof DataSampleScalarFieldEnum)[keyof typeof DataSampleScalarFieldEnum]


  export const ImageCaptureScalarFieldEnum: {
    id: 'id',
    sampleId: 'sampleId',
    fileName: 'fileName',
    imageUrl: 'imageUrl',
    createdAt: 'createdAt'
  };

  export type ImageCaptureScalarFieldEnum = (typeof ImageCaptureScalarFieldEnum)[keyof typeof ImageCaptureScalarFieldEnum]


  export const SortOrder: {
    asc: 'asc',
    desc: 'desc'
  };

  export type SortOrder = (typeof SortOrder)[keyof typeof SortOrder]


  export const QueryMode: {
    default: 'default',
    insensitive: 'insensitive'
  };

  export type QueryMode = (typeof QueryMode)[keyof typeof QueryMode]


  export const NullsOrder: {
    first: 'first',
    last: 'last'
  };

  export type NullsOrder = (typeof NullsOrder)[keyof typeof NullsOrder]


  /**
   * Field references
   */


  /**
   * Reference to a field of type 'Int'
   */
  export type IntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int'>
    


  /**
   * Reference to a field of type 'Int[]'
   */
  export type ListIntFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Int[]'>
    


  /**
   * Reference to a field of type 'String'
   */
  export type StringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String'>
    


  /**
   * Reference to a field of type 'String[]'
   */
  export type ListStringFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'String[]'>
    


  /**
   * Reference to a field of type 'DateTime'
   */
  export type DateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime'>
    


  /**
   * Reference to a field of type 'DateTime[]'
   */
  export type ListDateTimeFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'DateTime[]'>
    


  /**
   * Reference to a field of type 'Float'
   */
  export type FloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float'>
    


  /**
   * Reference to a field of type 'Float[]'
   */
  export type ListFloatFieldRefInput<$PrismaModel> = FieldRefInputType<$PrismaModel, 'Float[]'>
    
  /**
   * Deep Input Types
   */


  export type UserWhereInput = {
    AND?: UserWhereInput | UserWhereInput[]
    OR?: UserWhereInput[]
    NOT?: UserWhereInput | UserWhereInput[]
    id?: IntFilter<"User"> | number
    email?: StringFilter<"User"> | string
    password?: StringFilter<"User"> | string
    name?: StringNullableFilter<"User"> | string | null
    createdAt?: DateTimeFilter<"User"> | Date | string
  }

  export type UserOrderByWithRelationInput = {
    id?: SortOrder
    email?: SortOrder
    password?: SortOrder
    name?: SortOrderInput | SortOrder
    createdAt?: SortOrder
  }

  export type UserWhereUniqueInput = Prisma.AtLeast<{
    id?: number
    email?: string
    AND?: UserWhereInput | UserWhereInput[]
    OR?: UserWhereInput[]
    NOT?: UserWhereInput | UserWhereInput[]
    password?: StringFilter<"User"> | string
    name?: StringNullableFilter<"User"> | string | null
    createdAt?: DateTimeFilter<"User"> | Date | string
  }, "id" | "email">

  export type UserOrderByWithAggregationInput = {
    id?: SortOrder
    email?: SortOrder
    password?: SortOrder
    name?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    _count?: UserCountOrderByAggregateInput
    _avg?: UserAvgOrderByAggregateInput
    _max?: UserMaxOrderByAggregateInput
    _min?: UserMinOrderByAggregateInput
    _sum?: UserSumOrderByAggregateInput
  }

  export type UserScalarWhereWithAggregatesInput = {
    AND?: UserScalarWhereWithAggregatesInput | UserScalarWhereWithAggregatesInput[]
    OR?: UserScalarWhereWithAggregatesInput[]
    NOT?: UserScalarWhereWithAggregatesInput | UserScalarWhereWithAggregatesInput[]
    id?: IntWithAggregatesFilter<"User"> | number
    email?: StringWithAggregatesFilter<"User"> | string
    password?: StringWithAggregatesFilter<"User"> | string
    name?: StringNullableWithAggregatesFilter<"User"> | string | null
    createdAt?: DateTimeWithAggregatesFilter<"User"> | Date | string
  }

  export type DataBatchWhereInput = {
    AND?: DataBatchWhereInput | DataBatchWhereInput[]
    OR?: DataBatchWhereInput[]
    NOT?: DataBatchWhereInput | DataBatchWhereInput[]
    id?: IntFilter<"DataBatch"> | number
    name?: StringFilter<"DataBatch"> | string
    createdAt?: DateTimeFilter<"DataBatch"> | Date | string
    samples?: DataSampleListRelationFilter
  }

  export type DataBatchOrderByWithRelationInput = {
    id?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    samples?: DataSampleOrderByRelationAggregateInput
  }

  export type DataBatchWhereUniqueInput = Prisma.AtLeast<{
    id?: number
    AND?: DataBatchWhereInput | DataBatchWhereInput[]
    OR?: DataBatchWhereInput[]
    NOT?: DataBatchWhereInput | DataBatchWhereInput[]
    name?: StringFilter<"DataBatch"> | string
    createdAt?: DateTimeFilter<"DataBatch"> | Date | string
    samples?: DataSampleListRelationFilter
  }, "id">

  export type DataBatchOrderByWithAggregationInput = {
    id?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
    _count?: DataBatchCountOrderByAggregateInput
    _avg?: DataBatchAvgOrderByAggregateInput
    _max?: DataBatchMaxOrderByAggregateInput
    _min?: DataBatchMinOrderByAggregateInput
    _sum?: DataBatchSumOrderByAggregateInput
  }

  export type DataBatchScalarWhereWithAggregatesInput = {
    AND?: DataBatchScalarWhereWithAggregatesInput | DataBatchScalarWhereWithAggregatesInput[]
    OR?: DataBatchScalarWhereWithAggregatesInput[]
    NOT?: DataBatchScalarWhereWithAggregatesInput | DataBatchScalarWhereWithAggregatesInput[]
    id?: IntWithAggregatesFilter<"DataBatch"> | number
    name?: StringWithAggregatesFilter<"DataBatch"> | string
    createdAt?: DateTimeWithAggregatesFilter<"DataBatch"> | Date | string
  }

  export type DataSampleWhereInput = {
    AND?: DataSampleWhereInput | DataSampleWhereInput[]
    OR?: DataSampleWhereInput[]
    NOT?: DataSampleWhereInput | DataSampleWhereInput[]
    id?: IntFilter<"DataSample"> | number
    batchId?: IntFilter<"DataSample"> | number
    image_capture?: StringFilter<"DataSample"> | string
    classification?: StringFilter<"DataSample"> | string
    luster_value?: FloatNullableFilter<"DataSample"> | number | null
    roughness?: FloatNullableFilter<"DataSample"> | number | null
    tensile_strength?: FloatNullableFilter<"DataSample"> | number | null
    createdAt?: DateTimeFilter<"DataSample"> | Date | string
    batch?: XOR<DataBatchScalarRelationFilter, DataBatchWhereInput>
    images?: ImageCaptureListRelationFilter
  }

  export type DataSampleOrderByWithRelationInput = {
    id?: SortOrder
    batchId?: SortOrder
    image_capture?: SortOrder
    classification?: SortOrder
    luster_value?: SortOrderInput | SortOrder
    roughness?: SortOrderInput | SortOrder
    tensile_strength?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    batch?: DataBatchOrderByWithRelationInput
    images?: ImageCaptureOrderByRelationAggregateInput
  }

  export type DataSampleWhereUniqueInput = Prisma.AtLeast<{
    id?: number
    AND?: DataSampleWhereInput | DataSampleWhereInput[]
    OR?: DataSampleWhereInput[]
    NOT?: DataSampleWhereInput | DataSampleWhereInput[]
    batchId?: IntFilter<"DataSample"> | number
    image_capture?: StringFilter<"DataSample"> | string
    classification?: StringFilter<"DataSample"> | string
    luster_value?: FloatNullableFilter<"DataSample"> | number | null
    roughness?: FloatNullableFilter<"DataSample"> | number | null
    tensile_strength?: FloatNullableFilter<"DataSample"> | number | null
    createdAt?: DateTimeFilter<"DataSample"> | Date | string
    batch?: XOR<DataBatchScalarRelationFilter, DataBatchWhereInput>
    images?: ImageCaptureListRelationFilter
  }, "id">

  export type DataSampleOrderByWithAggregationInput = {
    id?: SortOrder
    batchId?: SortOrder
    image_capture?: SortOrder
    classification?: SortOrder
    luster_value?: SortOrderInput | SortOrder
    roughness?: SortOrderInput | SortOrder
    tensile_strength?: SortOrderInput | SortOrder
    createdAt?: SortOrder
    _count?: DataSampleCountOrderByAggregateInput
    _avg?: DataSampleAvgOrderByAggregateInput
    _max?: DataSampleMaxOrderByAggregateInput
    _min?: DataSampleMinOrderByAggregateInput
    _sum?: DataSampleSumOrderByAggregateInput
  }

  export type DataSampleScalarWhereWithAggregatesInput = {
    AND?: DataSampleScalarWhereWithAggregatesInput | DataSampleScalarWhereWithAggregatesInput[]
    OR?: DataSampleScalarWhereWithAggregatesInput[]
    NOT?: DataSampleScalarWhereWithAggregatesInput | DataSampleScalarWhereWithAggregatesInput[]
    id?: IntWithAggregatesFilter<"DataSample"> | number
    batchId?: IntWithAggregatesFilter<"DataSample"> | number
    image_capture?: StringWithAggregatesFilter<"DataSample"> | string
    classification?: StringWithAggregatesFilter<"DataSample"> | string
    luster_value?: FloatNullableWithAggregatesFilter<"DataSample"> | number | null
    roughness?: FloatNullableWithAggregatesFilter<"DataSample"> | number | null
    tensile_strength?: FloatNullableWithAggregatesFilter<"DataSample"> | number | null
    createdAt?: DateTimeWithAggregatesFilter<"DataSample"> | Date | string
  }

  export type ImageCaptureWhereInput = {
    AND?: ImageCaptureWhereInput | ImageCaptureWhereInput[]
    OR?: ImageCaptureWhereInput[]
    NOT?: ImageCaptureWhereInput | ImageCaptureWhereInput[]
    id?: IntFilter<"ImageCapture"> | number
    sampleId?: IntFilter<"ImageCapture"> | number
    fileName?: StringNullableFilter<"ImageCapture"> | string | null
    imageUrl?: StringFilter<"ImageCapture"> | string
    createdAt?: DateTimeFilter<"ImageCapture"> | Date | string
    sample?: XOR<DataSampleScalarRelationFilter, DataSampleWhereInput>
  }

  export type ImageCaptureOrderByWithRelationInput = {
    id?: SortOrder
    sampleId?: SortOrder
    fileName?: SortOrderInput | SortOrder
    imageUrl?: SortOrder
    createdAt?: SortOrder
    sample?: DataSampleOrderByWithRelationInput
  }

  export type ImageCaptureWhereUniqueInput = Prisma.AtLeast<{
    id?: number
    AND?: ImageCaptureWhereInput | ImageCaptureWhereInput[]
    OR?: ImageCaptureWhereInput[]
    NOT?: ImageCaptureWhereInput | ImageCaptureWhereInput[]
    sampleId?: IntFilter<"ImageCapture"> | number
    fileName?: StringNullableFilter<"ImageCapture"> | string | null
    imageUrl?: StringFilter<"ImageCapture"> | string
    createdAt?: DateTimeFilter<"ImageCapture"> | Date | string
    sample?: XOR<DataSampleScalarRelationFilter, DataSampleWhereInput>
  }, "id">

  export type ImageCaptureOrderByWithAggregationInput = {
    id?: SortOrder
    sampleId?: SortOrder
    fileName?: SortOrderInput | SortOrder
    imageUrl?: SortOrder
    createdAt?: SortOrder
    _count?: ImageCaptureCountOrderByAggregateInput
    _avg?: ImageCaptureAvgOrderByAggregateInput
    _max?: ImageCaptureMaxOrderByAggregateInput
    _min?: ImageCaptureMinOrderByAggregateInput
    _sum?: ImageCaptureSumOrderByAggregateInput
  }

  export type ImageCaptureScalarWhereWithAggregatesInput = {
    AND?: ImageCaptureScalarWhereWithAggregatesInput | ImageCaptureScalarWhereWithAggregatesInput[]
    OR?: ImageCaptureScalarWhereWithAggregatesInput[]
    NOT?: ImageCaptureScalarWhereWithAggregatesInput | ImageCaptureScalarWhereWithAggregatesInput[]
    id?: IntWithAggregatesFilter<"ImageCapture"> | number
    sampleId?: IntWithAggregatesFilter<"ImageCapture"> | number
    fileName?: StringNullableWithAggregatesFilter<"ImageCapture"> | string | null
    imageUrl?: StringWithAggregatesFilter<"ImageCapture"> | string
    createdAt?: DateTimeWithAggregatesFilter<"ImageCapture"> | Date | string
  }

  export type UserCreateInput = {
    email: string
    password: string
    name?: string | null
    createdAt?: Date | string
  }

  export type UserUncheckedCreateInput = {
    id?: number
    email: string
    password: string
    name?: string | null
    createdAt?: Date | string
  }

  export type UserUpdateInput = {
    email?: StringFieldUpdateOperationsInput | string
    password?: StringFieldUpdateOperationsInput | string
    name?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserUncheckedUpdateInput = {
    id?: IntFieldUpdateOperationsInput | number
    email?: StringFieldUpdateOperationsInput | string
    password?: StringFieldUpdateOperationsInput | string
    name?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserCreateManyInput = {
    id?: number
    email: string
    password: string
    name?: string | null
    createdAt?: Date | string
  }

  export type UserUpdateManyMutationInput = {
    email?: StringFieldUpdateOperationsInput | string
    password?: StringFieldUpdateOperationsInput | string
    name?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type UserUncheckedUpdateManyInput = {
    id?: IntFieldUpdateOperationsInput | number
    email?: StringFieldUpdateOperationsInput | string
    password?: StringFieldUpdateOperationsInput | string
    name?: NullableStringFieldUpdateOperationsInput | string | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataBatchCreateInput = {
    name: string
    createdAt?: Date | string
    samples?: DataSampleCreateNestedManyWithoutBatchInput
  }

  export type DataBatchUncheckedCreateInput = {
    id?: number
    name: string
    createdAt?: Date | string
    samples?: DataSampleUncheckedCreateNestedManyWithoutBatchInput
  }

  export type DataBatchUpdateInput = {
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    samples?: DataSampleUpdateManyWithoutBatchNestedInput
  }

  export type DataBatchUncheckedUpdateInput = {
    id?: IntFieldUpdateOperationsInput | number
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    samples?: DataSampleUncheckedUpdateManyWithoutBatchNestedInput
  }

  export type DataBatchCreateManyInput = {
    id?: number
    name: string
    createdAt?: Date | string
  }

  export type DataBatchUpdateManyMutationInput = {
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataBatchUncheckedUpdateManyInput = {
    id?: IntFieldUpdateOperationsInput | number
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataSampleCreateInput = {
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
    batch: DataBatchCreateNestedOneWithoutSamplesInput
    images?: ImageCaptureCreateNestedManyWithoutSampleInput
  }

  export type DataSampleUncheckedCreateInput = {
    id?: number
    batchId: number
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
    images?: ImageCaptureUncheckedCreateNestedManyWithoutSampleInput
  }

  export type DataSampleUpdateInput = {
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    batch?: DataBatchUpdateOneRequiredWithoutSamplesNestedInput
    images?: ImageCaptureUpdateManyWithoutSampleNestedInput
  }

  export type DataSampleUncheckedUpdateInput = {
    id?: IntFieldUpdateOperationsInput | number
    batchId?: IntFieldUpdateOperationsInput | number
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    images?: ImageCaptureUncheckedUpdateManyWithoutSampleNestedInput
  }

  export type DataSampleCreateManyInput = {
    id?: number
    batchId: number
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
  }

  export type DataSampleUpdateManyMutationInput = {
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataSampleUncheckedUpdateManyInput = {
    id?: IntFieldUpdateOperationsInput | number
    batchId?: IntFieldUpdateOperationsInput | number
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureCreateInput = {
    fileName?: string | null
    imageUrl: string
    createdAt?: Date | string
    sample: DataSampleCreateNestedOneWithoutImagesInput
  }

  export type ImageCaptureUncheckedCreateInput = {
    id?: number
    sampleId: number
    fileName?: string | null
    imageUrl: string
    createdAt?: Date | string
  }

  export type ImageCaptureUpdateInput = {
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    sample?: DataSampleUpdateOneRequiredWithoutImagesNestedInput
  }

  export type ImageCaptureUncheckedUpdateInput = {
    id?: IntFieldUpdateOperationsInput | number
    sampleId?: IntFieldUpdateOperationsInput | number
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureCreateManyInput = {
    id?: number
    sampleId: number
    fileName?: string | null
    imageUrl: string
    createdAt?: Date | string
  }

  export type ImageCaptureUpdateManyMutationInput = {
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureUncheckedUpdateManyInput = {
    id?: IntFieldUpdateOperationsInput | number
    sampleId?: IntFieldUpdateOperationsInput | number
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type IntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type StringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type StringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type DateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type SortOrderInput = {
    sort: SortOrder
    nulls?: NullsOrder
  }

  export type UserCountOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    password?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
  }

  export type UserAvgOrderByAggregateInput = {
    id?: SortOrder
  }

  export type UserMaxOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    password?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
  }

  export type UserMinOrderByAggregateInput = {
    id?: SortOrder
    email?: SortOrder
    password?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
  }

  export type UserSumOrderByAggregateInput = {
    id?: SortOrder
  }

  export type IntWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedIntFilter<$PrismaModel>
    _min?: NestedIntFilter<$PrismaModel>
    _max?: NestedIntFilter<$PrismaModel>
  }

  export type StringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type StringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    mode?: QueryMode
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type DateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type DataSampleListRelationFilter = {
    every?: DataSampleWhereInput
    some?: DataSampleWhereInput
    none?: DataSampleWhereInput
  }

  export type DataSampleOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type DataBatchCountOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
  }

  export type DataBatchAvgOrderByAggregateInput = {
    id?: SortOrder
  }

  export type DataBatchMaxOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
  }

  export type DataBatchMinOrderByAggregateInput = {
    id?: SortOrder
    name?: SortOrder
    createdAt?: SortOrder
  }

  export type DataBatchSumOrderByAggregateInput = {
    id?: SortOrder
  }

  export type FloatNullableFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableFilter<$PrismaModel> | number | null
  }

  export type DataBatchScalarRelationFilter = {
    is?: DataBatchWhereInput
    isNot?: DataBatchWhereInput
  }

  export type ImageCaptureListRelationFilter = {
    every?: ImageCaptureWhereInput
    some?: ImageCaptureWhereInput
    none?: ImageCaptureWhereInput
  }

  export type ImageCaptureOrderByRelationAggregateInput = {
    _count?: SortOrder
  }

  export type DataSampleCountOrderByAggregateInput = {
    id?: SortOrder
    batchId?: SortOrder
    image_capture?: SortOrder
    classification?: SortOrder
    luster_value?: SortOrder
    roughness?: SortOrder
    tensile_strength?: SortOrder
    createdAt?: SortOrder
  }

  export type DataSampleAvgOrderByAggregateInput = {
    id?: SortOrder
    batchId?: SortOrder
    luster_value?: SortOrder
    roughness?: SortOrder
    tensile_strength?: SortOrder
  }

  export type DataSampleMaxOrderByAggregateInput = {
    id?: SortOrder
    batchId?: SortOrder
    image_capture?: SortOrder
    classification?: SortOrder
    luster_value?: SortOrder
    roughness?: SortOrder
    tensile_strength?: SortOrder
    createdAt?: SortOrder
  }

  export type DataSampleMinOrderByAggregateInput = {
    id?: SortOrder
    batchId?: SortOrder
    image_capture?: SortOrder
    classification?: SortOrder
    luster_value?: SortOrder
    roughness?: SortOrder
    tensile_strength?: SortOrder
    createdAt?: SortOrder
  }

  export type DataSampleSumOrderByAggregateInput = {
    id?: SortOrder
    batchId?: SortOrder
    luster_value?: SortOrder
    roughness?: SortOrder
    tensile_strength?: SortOrder
  }

  export type FloatNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedFloatNullableFilter<$PrismaModel>
    _min?: NestedFloatNullableFilter<$PrismaModel>
    _max?: NestedFloatNullableFilter<$PrismaModel>
  }

  export type DataSampleScalarRelationFilter = {
    is?: DataSampleWhereInput
    isNot?: DataSampleWhereInput
  }

  export type ImageCaptureCountOrderByAggregateInput = {
    id?: SortOrder
    sampleId?: SortOrder
    fileName?: SortOrder
    imageUrl?: SortOrder
    createdAt?: SortOrder
  }

  export type ImageCaptureAvgOrderByAggregateInput = {
    id?: SortOrder
    sampleId?: SortOrder
  }

  export type ImageCaptureMaxOrderByAggregateInput = {
    id?: SortOrder
    sampleId?: SortOrder
    fileName?: SortOrder
    imageUrl?: SortOrder
    createdAt?: SortOrder
  }

  export type ImageCaptureMinOrderByAggregateInput = {
    id?: SortOrder
    sampleId?: SortOrder
    fileName?: SortOrder
    imageUrl?: SortOrder
    createdAt?: SortOrder
  }

  export type ImageCaptureSumOrderByAggregateInput = {
    id?: SortOrder
    sampleId?: SortOrder
  }

  export type StringFieldUpdateOperationsInput = {
    set?: string
  }

  export type NullableStringFieldUpdateOperationsInput = {
    set?: string | null
  }

  export type DateTimeFieldUpdateOperationsInput = {
    set?: Date | string
  }

  export type IntFieldUpdateOperationsInput = {
    set?: number
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type DataSampleCreateNestedManyWithoutBatchInput = {
    create?: XOR<DataSampleCreateWithoutBatchInput, DataSampleUncheckedCreateWithoutBatchInput> | DataSampleCreateWithoutBatchInput[] | DataSampleUncheckedCreateWithoutBatchInput[]
    connectOrCreate?: DataSampleCreateOrConnectWithoutBatchInput | DataSampleCreateOrConnectWithoutBatchInput[]
    createMany?: DataSampleCreateManyBatchInputEnvelope
    connect?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
  }

  export type DataSampleUncheckedCreateNestedManyWithoutBatchInput = {
    create?: XOR<DataSampleCreateWithoutBatchInput, DataSampleUncheckedCreateWithoutBatchInput> | DataSampleCreateWithoutBatchInput[] | DataSampleUncheckedCreateWithoutBatchInput[]
    connectOrCreate?: DataSampleCreateOrConnectWithoutBatchInput | DataSampleCreateOrConnectWithoutBatchInput[]
    createMany?: DataSampleCreateManyBatchInputEnvelope
    connect?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
  }

  export type DataSampleUpdateManyWithoutBatchNestedInput = {
    create?: XOR<DataSampleCreateWithoutBatchInput, DataSampleUncheckedCreateWithoutBatchInput> | DataSampleCreateWithoutBatchInput[] | DataSampleUncheckedCreateWithoutBatchInput[]
    connectOrCreate?: DataSampleCreateOrConnectWithoutBatchInput | DataSampleCreateOrConnectWithoutBatchInput[]
    upsert?: DataSampleUpsertWithWhereUniqueWithoutBatchInput | DataSampleUpsertWithWhereUniqueWithoutBatchInput[]
    createMany?: DataSampleCreateManyBatchInputEnvelope
    set?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    disconnect?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    delete?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    connect?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    update?: DataSampleUpdateWithWhereUniqueWithoutBatchInput | DataSampleUpdateWithWhereUniqueWithoutBatchInput[]
    updateMany?: DataSampleUpdateManyWithWhereWithoutBatchInput | DataSampleUpdateManyWithWhereWithoutBatchInput[]
    deleteMany?: DataSampleScalarWhereInput | DataSampleScalarWhereInput[]
  }

  export type DataSampleUncheckedUpdateManyWithoutBatchNestedInput = {
    create?: XOR<DataSampleCreateWithoutBatchInput, DataSampleUncheckedCreateWithoutBatchInput> | DataSampleCreateWithoutBatchInput[] | DataSampleUncheckedCreateWithoutBatchInput[]
    connectOrCreate?: DataSampleCreateOrConnectWithoutBatchInput | DataSampleCreateOrConnectWithoutBatchInput[]
    upsert?: DataSampleUpsertWithWhereUniqueWithoutBatchInput | DataSampleUpsertWithWhereUniqueWithoutBatchInput[]
    createMany?: DataSampleCreateManyBatchInputEnvelope
    set?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    disconnect?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    delete?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    connect?: DataSampleWhereUniqueInput | DataSampleWhereUniqueInput[]
    update?: DataSampleUpdateWithWhereUniqueWithoutBatchInput | DataSampleUpdateWithWhereUniqueWithoutBatchInput[]
    updateMany?: DataSampleUpdateManyWithWhereWithoutBatchInput | DataSampleUpdateManyWithWhereWithoutBatchInput[]
    deleteMany?: DataSampleScalarWhereInput | DataSampleScalarWhereInput[]
  }

  export type DataBatchCreateNestedOneWithoutSamplesInput = {
    create?: XOR<DataBatchCreateWithoutSamplesInput, DataBatchUncheckedCreateWithoutSamplesInput>
    connectOrCreate?: DataBatchCreateOrConnectWithoutSamplesInput
    connect?: DataBatchWhereUniqueInput
  }

  export type ImageCaptureCreateNestedManyWithoutSampleInput = {
    create?: XOR<ImageCaptureCreateWithoutSampleInput, ImageCaptureUncheckedCreateWithoutSampleInput> | ImageCaptureCreateWithoutSampleInput[] | ImageCaptureUncheckedCreateWithoutSampleInput[]
    connectOrCreate?: ImageCaptureCreateOrConnectWithoutSampleInput | ImageCaptureCreateOrConnectWithoutSampleInput[]
    createMany?: ImageCaptureCreateManySampleInputEnvelope
    connect?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
  }

  export type ImageCaptureUncheckedCreateNestedManyWithoutSampleInput = {
    create?: XOR<ImageCaptureCreateWithoutSampleInput, ImageCaptureUncheckedCreateWithoutSampleInput> | ImageCaptureCreateWithoutSampleInput[] | ImageCaptureUncheckedCreateWithoutSampleInput[]
    connectOrCreate?: ImageCaptureCreateOrConnectWithoutSampleInput | ImageCaptureCreateOrConnectWithoutSampleInput[]
    createMany?: ImageCaptureCreateManySampleInputEnvelope
    connect?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
  }

  export type NullableFloatFieldUpdateOperationsInput = {
    set?: number | null
    increment?: number
    decrement?: number
    multiply?: number
    divide?: number
  }

  export type DataBatchUpdateOneRequiredWithoutSamplesNestedInput = {
    create?: XOR<DataBatchCreateWithoutSamplesInput, DataBatchUncheckedCreateWithoutSamplesInput>
    connectOrCreate?: DataBatchCreateOrConnectWithoutSamplesInput
    upsert?: DataBatchUpsertWithoutSamplesInput
    connect?: DataBatchWhereUniqueInput
    update?: XOR<XOR<DataBatchUpdateToOneWithWhereWithoutSamplesInput, DataBatchUpdateWithoutSamplesInput>, DataBatchUncheckedUpdateWithoutSamplesInput>
  }

  export type ImageCaptureUpdateManyWithoutSampleNestedInput = {
    create?: XOR<ImageCaptureCreateWithoutSampleInput, ImageCaptureUncheckedCreateWithoutSampleInput> | ImageCaptureCreateWithoutSampleInput[] | ImageCaptureUncheckedCreateWithoutSampleInput[]
    connectOrCreate?: ImageCaptureCreateOrConnectWithoutSampleInput | ImageCaptureCreateOrConnectWithoutSampleInput[]
    upsert?: ImageCaptureUpsertWithWhereUniqueWithoutSampleInput | ImageCaptureUpsertWithWhereUniqueWithoutSampleInput[]
    createMany?: ImageCaptureCreateManySampleInputEnvelope
    set?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    disconnect?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    delete?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    connect?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    update?: ImageCaptureUpdateWithWhereUniqueWithoutSampleInput | ImageCaptureUpdateWithWhereUniqueWithoutSampleInput[]
    updateMany?: ImageCaptureUpdateManyWithWhereWithoutSampleInput | ImageCaptureUpdateManyWithWhereWithoutSampleInput[]
    deleteMany?: ImageCaptureScalarWhereInput | ImageCaptureScalarWhereInput[]
  }

  export type ImageCaptureUncheckedUpdateManyWithoutSampleNestedInput = {
    create?: XOR<ImageCaptureCreateWithoutSampleInput, ImageCaptureUncheckedCreateWithoutSampleInput> | ImageCaptureCreateWithoutSampleInput[] | ImageCaptureUncheckedCreateWithoutSampleInput[]
    connectOrCreate?: ImageCaptureCreateOrConnectWithoutSampleInput | ImageCaptureCreateOrConnectWithoutSampleInput[]
    upsert?: ImageCaptureUpsertWithWhereUniqueWithoutSampleInput | ImageCaptureUpsertWithWhereUniqueWithoutSampleInput[]
    createMany?: ImageCaptureCreateManySampleInputEnvelope
    set?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    disconnect?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    delete?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    connect?: ImageCaptureWhereUniqueInput | ImageCaptureWhereUniqueInput[]
    update?: ImageCaptureUpdateWithWhereUniqueWithoutSampleInput | ImageCaptureUpdateWithWhereUniqueWithoutSampleInput[]
    updateMany?: ImageCaptureUpdateManyWithWhereWithoutSampleInput | ImageCaptureUpdateManyWithWhereWithoutSampleInput[]
    deleteMany?: ImageCaptureScalarWhereInput | ImageCaptureScalarWhereInput[]
  }

  export type DataSampleCreateNestedOneWithoutImagesInput = {
    create?: XOR<DataSampleCreateWithoutImagesInput, DataSampleUncheckedCreateWithoutImagesInput>
    connectOrCreate?: DataSampleCreateOrConnectWithoutImagesInput
    connect?: DataSampleWhereUniqueInput
  }

  export type DataSampleUpdateOneRequiredWithoutImagesNestedInput = {
    create?: XOR<DataSampleCreateWithoutImagesInput, DataSampleUncheckedCreateWithoutImagesInput>
    connectOrCreate?: DataSampleCreateOrConnectWithoutImagesInput
    upsert?: DataSampleUpsertWithoutImagesInput
    connect?: DataSampleWhereUniqueInput
    update?: XOR<XOR<DataSampleUpdateToOneWithWhereWithoutImagesInput, DataSampleUpdateWithoutImagesInput>, DataSampleUncheckedUpdateWithoutImagesInput>
  }

  export type NestedIntFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntFilter<$PrismaModel> | number
  }

  export type NestedStringFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringFilter<$PrismaModel> | string
  }

  export type NestedStringNullableFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableFilter<$PrismaModel> | string | null
  }

  export type NestedDateTimeFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeFilter<$PrismaModel> | Date | string
  }

  export type NestedIntWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel>
    in?: number[] | ListIntFieldRefInput<$PrismaModel>
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel>
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntWithAggregatesFilter<$PrismaModel> | number
    _count?: NestedIntFilter<$PrismaModel>
    _avg?: NestedFloatFilter<$PrismaModel>
    _sum?: NestedIntFilter<$PrismaModel>
    _min?: NestedIntFilter<$PrismaModel>
    _max?: NestedIntFilter<$PrismaModel>
  }

  export type NestedFloatFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel>
    in?: number[] | ListFloatFieldRefInput<$PrismaModel>
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel>
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatFilter<$PrismaModel> | number
  }

  export type NestedStringWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel>
    in?: string[] | ListStringFieldRefInput<$PrismaModel>
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel>
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringWithAggregatesFilter<$PrismaModel> | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedStringFilter<$PrismaModel>
    _max?: NestedStringFilter<$PrismaModel>
  }

  export type NestedStringNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: string | StringFieldRefInput<$PrismaModel> | null
    in?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    notIn?: string[] | ListStringFieldRefInput<$PrismaModel> | null
    lt?: string | StringFieldRefInput<$PrismaModel>
    lte?: string | StringFieldRefInput<$PrismaModel>
    gt?: string | StringFieldRefInput<$PrismaModel>
    gte?: string | StringFieldRefInput<$PrismaModel>
    contains?: string | StringFieldRefInput<$PrismaModel>
    startsWith?: string | StringFieldRefInput<$PrismaModel>
    endsWith?: string | StringFieldRefInput<$PrismaModel>
    not?: NestedStringNullableWithAggregatesFilter<$PrismaModel> | string | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _min?: NestedStringNullableFilter<$PrismaModel>
    _max?: NestedStringNullableFilter<$PrismaModel>
  }

  export type NestedIntNullableFilter<$PrismaModel = never> = {
    equals?: number | IntFieldRefInput<$PrismaModel> | null
    in?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListIntFieldRefInput<$PrismaModel> | null
    lt?: number | IntFieldRefInput<$PrismaModel>
    lte?: number | IntFieldRefInput<$PrismaModel>
    gt?: number | IntFieldRefInput<$PrismaModel>
    gte?: number | IntFieldRefInput<$PrismaModel>
    not?: NestedIntNullableFilter<$PrismaModel> | number | null
  }

  export type NestedDateTimeWithAggregatesFilter<$PrismaModel = never> = {
    equals?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    in?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    notIn?: Date[] | string[] | ListDateTimeFieldRefInput<$PrismaModel>
    lt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    lte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gt?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    gte?: Date | string | DateTimeFieldRefInput<$PrismaModel>
    not?: NestedDateTimeWithAggregatesFilter<$PrismaModel> | Date | string
    _count?: NestedIntFilter<$PrismaModel>
    _min?: NestedDateTimeFilter<$PrismaModel>
    _max?: NestedDateTimeFilter<$PrismaModel>
  }

  export type NestedFloatNullableFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableFilter<$PrismaModel> | number | null
  }

  export type NestedFloatNullableWithAggregatesFilter<$PrismaModel = never> = {
    equals?: number | FloatFieldRefInput<$PrismaModel> | null
    in?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    notIn?: number[] | ListFloatFieldRefInput<$PrismaModel> | null
    lt?: number | FloatFieldRefInput<$PrismaModel>
    lte?: number | FloatFieldRefInput<$PrismaModel>
    gt?: number | FloatFieldRefInput<$PrismaModel>
    gte?: number | FloatFieldRefInput<$PrismaModel>
    not?: NestedFloatNullableWithAggregatesFilter<$PrismaModel> | number | null
    _count?: NestedIntNullableFilter<$PrismaModel>
    _avg?: NestedFloatNullableFilter<$PrismaModel>
    _sum?: NestedFloatNullableFilter<$PrismaModel>
    _min?: NestedFloatNullableFilter<$PrismaModel>
    _max?: NestedFloatNullableFilter<$PrismaModel>
  }

  export type DataSampleCreateWithoutBatchInput = {
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
    images?: ImageCaptureCreateNestedManyWithoutSampleInput
  }

  export type DataSampleUncheckedCreateWithoutBatchInput = {
    id?: number
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
    images?: ImageCaptureUncheckedCreateNestedManyWithoutSampleInput
  }

  export type DataSampleCreateOrConnectWithoutBatchInput = {
    where: DataSampleWhereUniqueInput
    create: XOR<DataSampleCreateWithoutBatchInput, DataSampleUncheckedCreateWithoutBatchInput>
  }

  export type DataSampleCreateManyBatchInputEnvelope = {
    data: DataSampleCreateManyBatchInput | DataSampleCreateManyBatchInput[]
    skipDuplicates?: boolean
  }

  export type DataSampleUpsertWithWhereUniqueWithoutBatchInput = {
    where: DataSampleWhereUniqueInput
    update: XOR<DataSampleUpdateWithoutBatchInput, DataSampleUncheckedUpdateWithoutBatchInput>
    create: XOR<DataSampleCreateWithoutBatchInput, DataSampleUncheckedCreateWithoutBatchInput>
  }

  export type DataSampleUpdateWithWhereUniqueWithoutBatchInput = {
    where: DataSampleWhereUniqueInput
    data: XOR<DataSampleUpdateWithoutBatchInput, DataSampleUncheckedUpdateWithoutBatchInput>
  }

  export type DataSampleUpdateManyWithWhereWithoutBatchInput = {
    where: DataSampleScalarWhereInput
    data: XOR<DataSampleUpdateManyMutationInput, DataSampleUncheckedUpdateManyWithoutBatchInput>
  }

  export type DataSampleScalarWhereInput = {
    AND?: DataSampleScalarWhereInput | DataSampleScalarWhereInput[]
    OR?: DataSampleScalarWhereInput[]
    NOT?: DataSampleScalarWhereInput | DataSampleScalarWhereInput[]
    id?: IntFilter<"DataSample"> | number
    batchId?: IntFilter<"DataSample"> | number
    image_capture?: StringFilter<"DataSample"> | string
    classification?: StringFilter<"DataSample"> | string
    luster_value?: FloatNullableFilter<"DataSample"> | number | null
    roughness?: FloatNullableFilter<"DataSample"> | number | null
    tensile_strength?: FloatNullableFilter<"DataSample"> | number | null
    createdAt?: DateTimeFilter<"DataSample"> | Date | string
  }

  export type DataBatchCreateWithoutSamplesInput = {
    name: string
    createdAt?: Date | string
  }

  export type DataBatchUncheckedCreateWithoutSamplesInput = {
    id?: number
    name: string
    createdAt?: Date | string
  }

  export type DataBatchCreateOrConnectWithoutSamplesInput = {
    where: DataBatchWhereUniqueInput
    create: XOR<DataBatchCreateWithoutSamplesInput, DataBatchUncheckedCreateWithoutSamplesInput>
  }

  export type ImageCaptureCreateWithoutSampleInput = {
    fileName?: string | null
    imageUrl: string
    createdAt?: Date | string
  }

  export type ImageCaptureUncheckedCreateWithoutSampleInput = {
    id?: number
    fileName?: string | null
    imageUrl: string
    createdAt?: Date | string
  }

  export type ImageCaptureCreateOrConnectWithoutSampleInput = {
    where: ImageCaptureWhereUniqueInput
    create: XOR<ImageCaptureCreateWithoutSampleInput, ImageCaptureUncheckedCreateWithoutSampleInput>
  }

  export type ImageCaptureCreateManySampleInputEnvelope = {
    data: ImageCaptureCreateManySampleInput | ImageCaptureCreateManySampleInput[]
    skipDuplicates?: boolean
  }

  export type DataBatchUpsertWithoutSamplesInput = {
    update: XOR<DataBatchUpdateWithoutSamplesInput, DataBatchUncheckedUpdateWithoutSamplesInput>
    create: XOR<DataBatchCreateWithoutSamplesInput, DataBatchUncheckedCreateWithoutSamplesInput>
    where?: DataBatchWhereInput
  }

  export type DataBatchUpdateToOneWithWhereWithoutSamplesInput = {
    where?: DataBatchWhereInput
    data: XOR<DataBatchUpdateWithoutSamplesInput, DataBatchUncheckedUpdateWithoutSamplesInput>
  }

  export type DataBatchUpdateWithoutSamplesInput = {
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataBatchUncheckedUpdateWithoutSamplesInput = {
    id?: IntFieldUpdateOperationsInput | number
    name?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureUpsertWithWhereUniqueWithoutSampleInput = {
    where: ImageCaptureWhereUniqueInput
    update: XOR<ImageCaptureUpdateWithoutSampleInput, ImageCaptureUncheckedUpdateWithoutSampleInput>
    create: XOR<ImageCaptureCreateWithoutSampleInput, ImageCaptureUncheckedCreateWithoutSampleInput>
  }

  export type ImageCaptureUpdateWithWhereUniqueWithoutSampleInput = {
    where: ImageCaptureWhereUniqueInput
    data: XOR<ImageCaptureUpdateWithoutSampleInput, ImageCaptureUncheckedUpdateWithoutSampleInput>
  }

  export type ImageCaptureUpdateManyWithWhereWithoutSampleInput = {
    where: ImageCaptureScalarWhereInput
    data: XOR<ImageCaptureUpdateManyMutationInput, ImageCaptureUncheckedUpdateManyWithoutSampleInput>
  }

  export type ImageCaptureScalarWhereInput = {
    AND?: ImageCaptureScalarWhereInput | ImageCaptureScalarWhereInput[]
    OR?: ImageCaptureScalarWhereInput[]
    NOT?: ImageCaptureScalarWhereInput | ImageCaptureScalarWhereInput[]
    id?: IntFilter<"ImageCapture"> | number
    sampleId?: IntFilter<"ImageCapture"> | number
    fileName?: StringNullableFilter<"ImageCapture"> | string | null
    imageUrl?: StringFilter<"ImageCapture"> | string
    createdAt?: DateTimeFilter<"ImageCapture"> | Date | string
  }

  export type DataSampleCreateWithoutImagesInput = {
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
    batch: DataBatchCreateNestedOneWithoutSamplesInput
  }

  export type DataSampleUncheckedCreateWithoutImagesInput = {
    id?: number
    batchId: number
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
  }

  export type DataSampleCreateOrConnectWithoutImagesInput = {
    where: DataSampleWhereUniqueInput
    create: XOR<DataSampleCreateWithoutImagesInput, DataSampleUncheckedCreateWithoutImagesInput>
  }

  export type DataSampleUpsertWithoutImagesInput = {
    update: XOR<DataSampleUpdateWithoutImagesInput, DataSampleUncheckedUpdateWithoutImagesInput>
    create: XOR<DataSampleCreateWithoutImagesInput, DataSampleUncheckedCreateWithoutImagesInput>
    where?: DataSampleWhereInput
  }

  export type DataSampleUpdateToOneWithWhereWithoutImagesInput = {
    where?: DataSampleWhereInput
    data: XOR<DataSampleUpdateWithoutImagesInput, DataSampleUncheckedUpdateWithoutImagesInput>
  }

  export type DataSampleUpdateWithoutImagesInput = {
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    batch?: DataBatchUpdateOneRequiredWithoutSamplesNestedInput
  }

  export type DataSampleUncheckedUpdateWithoutImagesInput = {
    id?: IntFieldUpdateOperationsInput | number
    batchId?: IntFieldUpdateOperationsInput | number
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type DataSampleCreateManyBatchInput = {
    id?: number
    image_capture: string
    classification: string
    luster_value?: number | null
    roughness?: number | null
    tensile_strength?: number | null
    createdAt?: Date | string
  }

  export type DataSampleUpdateWithoutBatchInput = {
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    images?: ImageCaptureUpdateManyWithoutSampleNestedInput
  }

  export type DataSampleUncheckedUpdateWithoutBatchInput = {
    id?: IntFieldUpdateOperationsInput | number
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
    images?: ImageCaptureUncheckedUpdateManyWithoutSampleNestedInput
  }

  export type DataSampleUncheckedUpdateManyWithoutBatchInput = {
    id?: IntFieldUpdateOperationsInput | number
    image_capture?: StringFieldUpdateOperationsInput | string
    classification?: StringFieldUpdateOperationsInput | string
    luster_value?: NullableFloatFieldUpdateOperationsInput | number | null
    roughness?: NullableFloatFieldUpdateOperationsInput | number | null
    tensile_strength?: NullableFloatFieldUpdateOperationsInput | number | null
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureCreateManySampleInput = {
    id?: number
    fileName?: string | null
    imageUrl: string
    createdAt?: Date | string
  }

  export type ImageCaptureUpdateWithoutSampleInput = {
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureUncheckedUpdateWithoutSampleInput = {
    id?: IntFieldUpdateOperationsInput | number
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }

  export type ImageCaptureUncheckedUpdateManyWithoutSampleInput = {
    id?: IntFieldUpdateOperationsInput | number
    fileName?: NullableStringFieldUpdateOperationsInput | string | null
    imageUrl?: StringFieldUpdateOperationsInput | string
    createdAt?: DateTimeFieldUpdateOperationsInput | Date | string
  }



  /**
   * Batch Payload for updateMany & deleteMany & createMany
   */

  export type BatchPayload = {
    count: number
  }

  /**
   * DMMF
   */
  export const dmmf: runtime.BaseDMMF
}